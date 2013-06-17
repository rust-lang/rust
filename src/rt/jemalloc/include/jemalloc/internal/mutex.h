/******************************************************************************/
#ifdef JEMALLOC_H_TYPES

typedef struct malloc_mutex_s malloc_mutex_t;

#ifdef _WIN32
#  define MALLOC_MUTEX_INITIALIZER
#elif (defined(JEMALLOC_OSSPIN))
#  define MALLOC_MUTEX_INITIALIZER {0}
#elif (defined(JEMALLOC_MUTEX_INIT_CB))
#  define MALLOC_MUTEX_INITIALIZER {PTHREAD_MUTEX_INITIALIZER, NULL}
#else
#  if (defined(PTHREAD_MUTEX_ADAPTIVE_NP) &&				\
       defined(PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP))
#    define MALLOC_MUTEX_TYPE PTHREAD_MUTEX_ADAPTIVE_NP
#    define MALLOC_MUTEX_INITIALIZER {PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP}
#  else
#    define MALLOC_MUTEX_TYPE PTHREAD_MUTEX_DEFAULT
#    define MALLOC_MUTEX_INITIALIZER {PTHREAD_MUTEX_INITIALIZER}
#  endif
#endif

#endif /* JEMALLOC_H_TYPES */
/******************************************************************************/
#ifdef JEMALLOC_H_STRUCTS

struct malloc_mutex_s {
#ifdef _WIN32
	CRITICAL_SECTION	lock;
#elif (defined(JEMALLOC_OSSPIN))
	OSSpinLock		lock;
#elif (defined(JEMALLOC_MUTEX_INIT_CB))
	pthread_mutex_t		lock;
	malloc_mutex_t		*postponed_next;
#else
	pthread_mutex_t		lock;
#endif
};

#endif /* JEMALLOC_H_STRUCTS */
/******************************************************************************/
#ifdef JEMALLOC_H_EXTERNS

#ifdef JEMALLOC_LAZY_LOCK
extern bool isthreaded;
#else
#  undef isthreaded /* Undo private_namespace.h definition. */
#  define isthreaded true
#endif

bool	malloc_mutex_init(malloc_mutex_t *mutex);
void	malloc_mutex_prefork(malloc_mutex_t *mutex);
void	malloc_mutex_postfork_parent(malloc_mutex_t *mutex);
void	malloc_mutex_postfork_child(malloc_mutex_t *mutex);
bool	mutex_boot(void);

#endif /* JEMALLOC_H_EXTERNS */
/******************************************************************************/
#ifdef JEMALLOC_H_INLINES

#ifndef JEMALLOC_ENABLE_INLINE
void	malloc_mutex_lock(malloc_mutex_t *mutex);
void	malloc_mutex_unlock(malloc_mutex_t *mutex);
#endif

#if (defined(JEMALLOC_ENABLE_INLINE) || defined(JEMALLOC_MUTEX_C_))
JEMALLOC_INLINE void
malloc_mutex_lock(malloc_mutex_t *mutex)
{

	if (isthreaded) {
#ifdef _WIN32
		EnterCriticalSection(&mutex->lock);
#elif (defined(JEMALLOC_OSSPIN))
		OSSpinLockLock(&mutex->lock);
#else
		pthread_mutex_lock(&mutex->lock);
#endif
	}
}

JEMALLOC_INLINE void
malloc_mutex_unlock(malloc_mutex_t *mutex)
{

	if (isthreaded) {
#ifdef _WIN32
		LeaveCriticalSection(&mutex->lock);
#elif (defined(JEMALLOC_OSSPIN))
		OSSpinLockUnlock(&mutex->lock);
#else
		pthread_mutex_unlock(&mutex->lock);
#endif
	}
}
#endif

#endif /* JEMALLOC_H_INLINES */
/******************************************************************************/
