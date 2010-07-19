#include "../globals.h"

/*
 * Conditional variable. Implemented using pthreads condition variables, and
 * using events on windows.
 */

#include "condition_variable.h"

// #define TRACE

condition_variable::condition_variable() {
#if defined(__WIN32__)
    _event = CreateEvent(NULL, FALSE, FALSE, NULL);
#else
    pthread_cond_init(&_cond, NULL);
    pthread_mutex_init(&_mutex, NULL);
#endif
}

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
#ifdef TRACE
    printf("waiting on condition_variable: 0x%" PRIxPTR "\n",
           (uintptr_t)this);
#endif
#if defined(__WIN32__)
    WaitForSingleObject(_event, INFINITE);
#else
    pthread_mutex_lock(&_mutex);
    pthread_cond_wait(&_cond, &_mutex);
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
