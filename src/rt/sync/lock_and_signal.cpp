// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#include "../rust_globals.h"
#include "lock_and_signal.h"

/*
 * A "lock-and-signal" pair. These are necessarily coupled on pthreads
 * systems, and artificially coupled (by this file) on win32. Put
 * together here to minimize ifdefs elsewhere; you must use them as
 * if you're using a pthreads cvar+mutex pair.
 */

// FIXME (#2683): This is not a portable way of specifying an invalid
// pthread_t
#define INVALID_THREAD 0


#if defined(__WIN32__)
lock_and_signal::lock_and_signal()
#if defined(DEBUG_LOCKS)
    : _holding_thread(INVALID_THREAD)
#endif
{
    _event = CreateEvent(NULL, FALSE, FALSE, NULL);

    // If a CRITICAL_SECTION is not initialized with a spin count, it will
    // default to 0, even on multi-processor systems. MSDN suggests using
    // 4000. On single-processor systems, the spin count parameter is ignored
    // and the critical section's spin count defaults to 0.
    const DWORD SPIN_COUNT = 4000;
    CHECKED(!InitializeCriticalSectionAndSpinCount(&_cs, SPIN_COUNT));

    // FIXME #2893 Consider checking
    // GetProcAddress("InitializeCriticalSectionEx")
    // so Windows >= Vista we can use CRITICAL_SECTION_NO_DEBUG_INFO to avoid
    // allocating CRITICAL_SECTION debug info that is never released. See:
    // http://stackoverflow.com/questions/804848/
    //        critical-sections-leaking-memory-on-vista-win2008#889853
}

#else
lock_and_signal::lock_and_signal()
#if defined(DEBUG_LOCKS)
    : _holding_thread(INVALID_THREAD)
#endif
{
    CHECKED(pthread_cond_init(&_cond, NULL));
    CHECKED(pthread_mutex_init(&_mutex, NULL));
}
#endif

lock_and_signal::~lock_and_signal() {
#if defined(__WIN32__)
    CloseHandle(_event);
    DeleteCriticalSection(&_cs);
#else
    CHECKED(pthread_cond_destroy(&_cond));
    CHECKED(pthread_mutex_destroy(&_mutex));
#endif
}

void lock_and_signal::lock() {
    must_not_have_lock();
#if defined(__WIN32__)
    EnterCriticalSection(&_cs);
#if defined(DEBUG_LOCKS)
    _holding_thread = GetCurrentThreadId();
#endif
#else
    CHECKED(pthread_mutex_lock(&_mutex));
#if defined(DEBUG_LOCKS)
    _holding_thread = pthread_self();
#endif
#endif
}

bool lock_and_signal::try_lock() {
    must_not_have_lock();
#if defined(__WIN32__)
    if (TryEnterCriticalSection(&_cs)) {
#if defined(DEBUG_LOCKS)
        _holding_thread = GetCurrentThreadId();
#endif
        return true;
    }
#else // non-windows
    int trylock = pthread_mutex_trylock(&_mutex);
    if (trylock == 0) {
#if defined(DEBUG_LOCKS)
        _holding_thread = pthread_self();
#endif
        return true;
    } else if (trylock == EBUSY) {
        // EBUSY means lock was already held by someone else
        return false;
    }
    // abort on all other errors
    CHECKED(trylock);
#endif
    return false;
}

void lock_and_signal::unlock() {
    must_have_lock();
#if defined(DEBUG_LOCKS)
    _holding_thread = INVALID_THREAD;
#endif
#if defined(__WIN32__)
    LeaveCriticalSection(&_cs);
#else
    CHECKED(pthread_mutex_unlock(&_mutex));
#endif
}

/**
 * Wait indefinitely until condition is signaled.
 */
void lock_and_signal::wait() {
    must_have_lock();
#if defined(DEBUG_LOCKS)
    _holding_thread = INVALID_THREAD;
#endif
#if defined(__WIN32__)
    LeaveCriticalSection(&_cs);
    WaitForSingleObject(_event, INFINITE);
    EnterCriticalSection(&_cs);
    must_not_be_locked();
#if defined(DEBUG_LOCKS)
    _holding_thread = GetCurrentThreadId();
#endif
#else
    CHECKED(pthread_cond_wait(&_cond, &_mutex));
    must_not_be_locked();
#if defined(DEBUG_LOCKS)
    _holding_thread = pthread_self();
#endif
#endif
}

/**
 * Signal condition, and resume the waiting thread.
 */
void lock_and_signal::signal() {
#if defined(__WIN32__)
    SetEvent(_event);
#else
    CHECKED(pthread_cond_signal(&_cond));
#endif
}

#if defined(DEBUG_LOCKS)
bool lock_and_signal::lock_held_by_current_thread()
{
#if defined(__WIN32__)
    return _holding_thread == GetCurrentThreadId();
#else
    return pthread_equal(_holding_thread, pthread_self());
#endif
}
#endif

#if defined(DEBUG_LOCKS)
void lock_and_signal::must_have_lock() {
    assert(lock_held_by_current_thread() && "must have lock");
}
void lock_and_signal::must_not_have_lock() {
    assert(!lock_held_by_current_thread() && "must not have lock");
}
void lock_and_signal::must_not_be_locked() {
}
#else
void lock_and_signal::must_have_lock() { }
void lock_and_signal::must_not_have_lock() { }
void lock_and_signal::must_not_be_locked() { }
#endif

scoped_lock::scoped_lock(lock_and_signal &lock)
    : lock(lock)
{
    lock.lock();
}

scoped_lock::~scoped_lock()
{
    lock.unlock();
}

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
