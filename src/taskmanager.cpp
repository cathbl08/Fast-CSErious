#include <chrono>
#include <thread>

// #include <concurrentqueue.h>
#include "../concurrentqueue/concurrentqueue.h"

#include "taskmanager.hpp"
#include "timer.hpp"


namespace ASC_HPC
{

  class Task
  {
  public:
    int nr, size;
    const std::function<void(int nr, int size)> * pfunc;
    std::atomic<int> * cnt;

    Task & operator++(int)
    {
      nr++;
      return *this;
    }
    Task & operator*() { return *this; }
  };

  
  typedef moodycamel::ConcurrentQueue<Task> TQueue; 
  typedef moodycamel::ProducerToken TPToken; 
  typedef moodycamel::ConsumerToken TCToken; 
  
  
  static std::atomic<bool> stop{false};
  static std::vector<std::thread> threads;
  static TQueue queue;
  
  void StartWorkers(int num)
  {
    stop = false;
    for (int i = 0; i < num; i++)
      {
        TimeLine * patl = timeline.get();
        threads.push_back
          (std::thread([patl]()
          {
            if (patl)
              timeline = std::make_unique<TimeLine>();
          
            TPToken ptoken(queue); 
            TCToken ctoken(queue); 
            
            while(true)
              {
                if (stop) break;

                Task task;
                if(!queue.try_dequeue_from_producer(ptoken, task)) 
                  if(!queue.try_dequeue(ctoken, task))  
                    continue; 
                
                /*Task task;
                if(!queue.try_dequeue_from_producer(ptoken, task)) {
                  if(!queue.try_dequeue(ctoken, task)) {
                    // The worker thread yields/sleeps if the queue is empty.
                    // This allows the OS to schedule CPU time to the main thread (Thread 0) 
                    // which is necessary for it to enqueue tasks.
                    std::this_thread::yield(); // Or std::this_thread::sleep_for(std::chrono::microseconds(1));
                    continue; 
                  }
                }*/
                
                (*task.pfunc)(task.nr, task.size);
                (*task.cnt)++;
              }
            
            if (patl)
              patl -> addTimeLine(std::move(*timeline));
          }));
      }
  }

  void StopWorkers()
  {
    stop = true;
    for (auto & t : threads)
      t.join();
    threads.clear();
  }

  
  void RunParallel (int num,
                    const std::function<void(int nr, int size)> & func)
  {
    TPToken ptoken(queue);
    TCToken ctoken(queue);
    
    std::atomic<int> cnt{0};


    for (size_t i = 0; i < num; i++)
      {
        Task task;
        task.nr = i;
        task.size = num;
        task.pfunc = &func;
        task.cnt = &cnt;
        queue.enqueue(ptoken, task);
      }

    /*
    // faster with bulk enqueue (error with gcc-Release)
    Task firsttask;
    firsttask.nr = 0;
    firsttask.size = num;
    firsttask.pfunc=&func;
    firsttask.cnt = &cnt;
    queue.enqueue_bulk (ptoken, firsttask, num);    
    */
    
    while (cnt < num)
      {
        Task task;
        if(!queue.try_dequeue_from_producer(ptoken, task)) 
          if(!queue.try_dequeue(ctoken, task))
            continue; 
        
        (*task.pfunc)(task.nr, task.size);
        (*task.cnt)++;
      }
    
    /*while (cnt < num)
      {
        Task task;
        if(!queue.try_dequeue_from_producer(ptoken, task)) 
          if(!queue.try_dequeue(ctoken, task))
            {
              // Yield control to the OS scheduler if the queue is empty.
              // This allows idle worker threads (Threads 1, 2, 3...) to run.
              std::this_thread::yield(); 
              continue;
            }
        
        (*task.pfunc)(task.nr, task.size);
        (*task.cnt)++;
      }*/
  }
}
