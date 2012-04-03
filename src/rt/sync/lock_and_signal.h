// -*- c++ -*-
#ifndef LOCK_AND_SIGNAL_H
#define LOCK_AND_SIGNAL_H

#include "rust_globals.h"

#ifndef RUST_NDEBUG
#define DEBUG_LOCKS
#endif

class lock_and_signal {
#if defined(__WIN32__)
    HANDLE _event;
    CRITICAL_SECTION _cs;
#if defined(DEBUG_LOCKS)
    DWORD _holding_thread;
#endif
#else
    pthread_cond_t _cond;
    pthread_mutex_t _mutex;
#if defined(DEBUG_LOCKS)
    pthread_t _holding_thread;
#endif
#endif

#if defined(DEBUG_LOCKS)
    bool lock_held_by_current_thread();
#endif

    void must_not_be_locked();

public:
    lock_and_signal();
    virtual ~lock_and_signal();

    void lock();
    void unlock();
    void wait();
    void signal();

    void must_have_lock();
    void must_not_have_lock();
};

class scoped_lock {
  lock_and_signal &lock;

public:
  scoped_lock(lock_and_signal &lock);
  ~scoped_lock();
};

#endif /* LOCK_AND_SIGNAL_H */
