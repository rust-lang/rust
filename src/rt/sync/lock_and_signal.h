// -*- c++ -*-
#ifndef LOCK_AND_SIGNAL_H
#define LOCK_AND_SIGNAL_H

class lock_and_signal {
#if defined(__WIN32__)
    HANDLE _event;
    CRITICAL_SECTION _cs;
    DWORD _holding_thread;
#else
    pthread_cond_t _cond;
    pthread_mutex_t _mutex;

    pthread_t _holding_thread;
#endif

public:
    lock_and_signal();
    virtual ~lock_and_signal();

    void lock();
    void unlock();
    void wait();
    void signal();

    bool lock_held_by_current_thread();
};

class scoped_lock {
  lock_and_signal &lock;

public:
  scoped_lock(lock_and_signal &lock);
  ~scoped_lock();
};

#endif /* LOCK_AND_SIGNAL_H */
