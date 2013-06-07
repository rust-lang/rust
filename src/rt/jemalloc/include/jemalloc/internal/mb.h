/******************************************************************************/
#ifdef JEMALLOC_H_TYPES

#endif /* JEMALLOC_H_TYPES */
/******************************************************************************/
#ifdef JEMALLOC_H_STRUCTS

#endif /* JEMALLOC_H_STRUCTS */
/******************************************************************************/
#ifdef JEMALLOC_H_EXTERNS

#endif /* JEMALLOC_H_EXTERNS */
/******************************************************************************/
#ifdef JEMALLOC_H_INLINES

#ifndef JEMALLOC_ENABLE_INLINE
void	mb_write(void);
#endif

#if (defined(JEMALLOC_ENABLE_INLINE) || defined(JEMALLOC_MB_C_))
#ifdef __i386__
/*
 * According to the Intel Architecture Software Developer's Manual, current
 * processors execute instructions in order from the perspective of other
 * processors in a multiprocessor system, but 1) Intel reserves the right to
 * change that, and 2) the compiler's optimizer could re-order instructions if
 * there weren't some form of barrier.  Therefore, even if running on an
 * architecture that does not need memory barriers (everything through at least
 * i686), an "optimizer barrier" is necessary.
 */
JEMALLOC_INLINE void
mb_write(void)
{

#  if 0
    /* This is a true memory barrier. */
    asm volatile ("pusha;"
        "xor  %%eax,%%eax;"
        "cpuid;"
        "popa;"
        : /* Outputs. */
        : /* Inputs. */
        : "memory" /* Clobbers. */
        );
#else
    /*
     * This is hopefully enough to keep the compiler from reordering
     * instructions around this one.
     */
    asm volatile ("nop;"
        : /* Outputs. */
        : /* Inputs. */
        : "memory" /* Clobbers. */
        );
#endif
}
#elif (defined(__amd64__) || defined(__x86_64__))
JEMALLOC_INLINE void
mb_write(void)
{

    asm volatile ("sfence"
        : /* Outputs. */
        : /* Inputs. */
        : "memory" /* Clobbers. */
        );
}
#elif defined(__powerpc__)
JEMALLOC_INLINE void
mb_write(void)
{

    asm volatile ("eieio"
        : /* Outputs. */
        : /* Inputs. */
        : "memory" /* Clobbers. */
        );
}
#elif defined(__sparc64__)
JEMALLOC_INLINE void
mb_write(void)
{

    asm volatile ("membar #StoreStore"
        : /* Outputs. */
        : /* Inputs. */
        : "memory" /* Clobbers. */
        );
}
#elif defined(__tile__)
JEMALLOC_INLINE void
mb_write(void)
{

    __sync_synchronize();
}
#else
/*
 * This is much slower than a simple memory barrier, but the semantics of mutex
 * unlock make this work.
 */
JEMALLOC_INLINE void
mb_write(void)
{
    malloc_mutex_t mtx;

    malloc_mutex_init(&mtx);
    malloc_mutex_lock(&mtx);
    malloc_mutex_unlock(&mtx);
}
#endif
#endif

#endif /* JEMALLOC_H_INLINES */
/******************************************************************************/
