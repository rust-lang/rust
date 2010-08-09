#include "../globals.h"

/*
 * Conditional variable. Implemented using pthreads condition variables, and
 * using events on windows.
 */

#include "condition_variable.h"

// #define TRACE

#if defined(__WIN32__)
condition_variable::condition_variable() {
    _event = CreateEvent(NULL, FALSE, FALSE, NULL);
}
#else
condition_variable::condition_variable() {
    pthread_cond_init(&_cond, NULL);
    pthread_mutex_init(&_mutex, NULL);
}
#endif

condition_variable::~condition_variable() {
#if defined(__WIN32__)
    CloseHandle(_event);
#else
    pthread_cond_destroy(&_cond);
    pthread_mutex_destroy(&_mutex);
#endif
}

/**
 * Wait indefinitely until condition is signaled.
 */
void condition_variable::wait() {
    timed_wait(0);
}

void condition_variable::timed_wait(size_t timeout_in_ns) {
#ifdef TRACE
    printf("waiting on condition_variable: 0x%" PRIxPTR " for %d ns. \n",
           (uintptr_t) this, (int) timeout_in_ns);
#endif
#if defined(__WIN32__)
    WaitForSingleObject(_event, INFINITE);
#else
    pthread_mutex_lock(&_mutex);
    // wait() automatically releases the mutex while it waits, and acquires
    // it right before exiting. This allows signal() to acquire the mutex
    // when signaling.)
    if (timeout_in_ns == 0) {
        pthread_cond_wait(&_cond, &_mutex);
    } else {
        timeval time_val;
        gettimeofday(&time_val, NULL);
        timespec time_spec;
        time_spec.tv_sec = time_val.tv_sec + 0;
        time_spec.tv_nsec = time_val.tv_usec * 1000 + timeout_in_ns;
        pthread_cond_timedwait(&_cond, &_mutex, &time_spec);
    }
    pthread_mutex_unlock(&_mutex);
#endif
#ifdef TRACE
    printf("resumed on condition_variable: 0x%" PRIxPTR "\n",
           (uintptr_t)this);
#endif
}

/**
 * Signal condition, and resume the waiting thread.
 */
void condition_variable::signal() {
#if defined(__WIN32__)
    SetEvent(_event);
#else
    pthread_mutex_lock(&_mutex);
    pthread_cond_signal(&_cond);
    pthread_mutex_unlock(&_mutex);
#endif
#ifdef TRACE
    printf("signal  condition_variable: 0x%" PRIxPTR "\n",
           (uintptr_t)this);
#endif
}
