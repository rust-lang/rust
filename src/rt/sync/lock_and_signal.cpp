#include <assert.h>
#include "../globals.h"

/*
 * A "lock-and-signal" pair. These are necessarily coupled on pthreads
 * systems, and artificially coupled (by this file) on win32. Put
 * together here to minimize ifdefs elsewhere; you must use them as
 * if you're using a pthreads cvar+mutex pair.
 */

#include "lock_and_signal.h"

// FIXME: This is not a portable way of specifying an invalid pthread_t
#define INVALID_THREAD 0


#if defined(__WIN32__)
lock_and_signal::lock_and_signal()
    : _holding_thread(INVALID_THREAD)
{
    _event = CreateEvent(NULL, TRUE, FALSE, NULL);
    InitializeCriticalSection(&_cs);
}

#else
lock_and_signal::lock_and_signal()
    : _holding_thread(INVALID_THREAD)
{
    CHECKED(pthread_cond_init(&_cond, NULL));
    CHECKED(pthread_mutex_init(&_mutex, NULL));
}
#endif

lock_and_signal::~lock_and_signal() {
#if defined(__WIN32__)
    CloseHandle(_event);
#else
    CHECKED(pthread_cond_destroy(&_cond));
    CHECKED(pthread_mutex_destroy(&_mutex));
#endif
}

void lock_and_signal::lock() {
#if defined(__WIN32__)
    EnterCriticalSection(&_cs);
    _holding_thread = GetCurrentThreadId();
#else
    CHECKED(pthread_mutex_lock(&_mutex));
    _holding_thread = pthread_self();
#endif
}

void lock_and_signal::unlock() {
    _holding_thread = INVALID_THREAD;
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
    assert(lock_held_by_current_thread());
    _holding_thread = INVALID_THREAD;
#if defined(__WIN32__)
    LeaveCriticalSection(&_cs);
    WaitForSingleObject(_event, INFINITE);
    EnterCriticalSection(&_cs);
    _holding_thread = GetCurrentThreadId();
#else
    CHECKED(pthread_cond_wait(&_cond, &_mutex));
    _holding_thread = pthread_self();
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

bool lock_and_signal::lock_held_by_current_thread()
{
#if defined(__WIN32__)
    return _holding_thread == GetCurrentThreadId();
#else
    return pthread_equal(_holding_thread, pthread_self());
#endif
}

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
