/******************************************************************************/
#ifdef JEMALLOC_H_TYPES

#endif /* JEMALLOC_H_TYPES */
/******************************************************************************/
#ifdef JEMALLOC_H_STRUCTS

#endif /* JEMALLOC_H_STRUCTS */
/******************************************************************************/
#ifdef JEMALLOC_H_EXTERNS

#define	atomic_read_uint64(p)	atomic_add_uint64(p, 0)
#define	atomic_read_uint32(p)	atomic_add_uint32(p, 0)
#define	atomic_read_z(p)	atomic_add_z(p, 0)
#define	atomic_read_u(p)	atomic_add_u(p, 0)

#endif /* JEMALLOC_H_EXTERNS */
/******************************************************************************/
#ifdef JEMALLOC_H_INLINES

#ifndef JEMALLOC_ENABLE_INLINE
uint64_t	atomic_add_uint64(uint64_t *p, uint64_t x);
uint64_t	atomic_sub_uint64(uint64_t *p, uint64_t x);
uint32_t	atomic_add_uint32(uint32_t *p, uint32_t x);
uint32_t	atomic_sub_uint32(uint32_t *p, uint32_t x);
size_t	atomic_add_z(size_t *p, size_t x);
size_t	atomic_sub_z(size_t *p, size_t x);
unsigned	atomic_add_u(unsigned *p, unsigned x);
unsigned	atomic_sub_u(unsigned *p, unsigned x);
#endif

#if (defined(JEMALLOC_ENABLE_INLINE) || defined(JEMALLOC_ATOMIC_C_))
/******************************************************************************/
/* 64-bit operations. */
#if (LG_SIZEOF_PTR == 3 || LG_SIZEOF_INT == 3)
#  ifdef __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8
JEMALLOC_INLINE uint64_t
atomic_add_uint64(uint64_t *p, uint64_t x)
{

    return (__sync_add_and_fetch(p, x));
}

JEMALLOC_INLINE uint64_t
atomic_sub_uint64(uint64_t *p, uint64_t x)
{

    return (__sync_sub_and_fetch(p, x));
}
#elif (defined(_MSC_VER))
JEMALLOC_INLINE uint64_t
atomic_add_uint64(uint64_t *p, uint64_t x)
{

    return (InterlockedExchangeAdd64(p, x));
}

JEMALLOC_INLINE uint64_t
atomic_sub_uint64(uint64_t *p, uint64_t x)
{

    return (InterlockedExchangeAdd64(p, -((int64_t)x)));
}
#elif (defined(JEMALLOC_OSATOMIC))
JEMALLOC_INLINE uint64_t
atomic_add_uint64(uint64_t *p, uint64_t x)
{

    return (OSAtomicAdd64((int64_t)x, (int64_t *)p));
}

JEMALLOC_INLINE uint64_t
atomic_sub_uint64(uint64_t *p, uint64_t x)
{

    return (OSAtomicAdd64(-((int64_t)x), (int64_t *)p));
}
#  elif (defined(__amd64__) || defined(__x86_64__))
JEMALLOC_INLINE uint64_t
atomic_add_uint64(uint64_t *p, uint64_t x)
{

    asm volatile (
        "lock; xaddq %0, %1;"
        : "+r" (x), "=m" (*p) /* Outputs. */
        : "m" (*p) /* Inputs. */
        );

    return (x);
}

JEMALLOC_INLINE uint64_t
atomic_sub_uint64(uint64_t *p, uint64_t x)
{

    x = (uint64_t)(-(int64_t)x);
    asm volatile (
        "lock; xaddq %0, %1;"
        : "+r" (x), "=m" (*p) /* Outputs. */
        : "m" (*p) /* Inputs. */
        );

    return (x);
}
#  elif (defined(JEMALLOC_ATOMIC9))
JEMALLOC_INLINE uint64_t
atomic_add_uint64(uint64_t *p, uint64_t x)
{

    /*
     * atomic_fetchadd_64() doesn't exist, but we only ever use this
     * function on LP64 systems, so atomic_fetchadd_long() will do.
     */
    assert(sizeof(uint64_t) == sizeof(unsigned long));

    return (atomic_fetchadd_long(p, (unsigned long)x) + x);
}

JEMALLOC_INLINE uint64_t
atomic_sub_uint64(uint64_t *p, uint64_t x)
{

    assert(sizeof(uint64_t) == sizeof(unsigned long));

    return (atomic_fetchadd_long(p, (unsigned long)(-(long)x)) - x);
}
#  elif (defined(JE_FORCE_SYNC_COMPARE_AND_SWAP_8))
JEMALLOC_INLINE uint64_t
atomic_add_uint64(uint64_t *p, uint64_t x)
{

    return (__sync_add_and_fetch(p, x));
}

JEMALLOC_INLINE uint64_t
atomic_sub_uint64(uint64_t *p, uint64_t x)
{

    return (__sync_sub_and_fetch(p, x));
}
#  else
#    error "Missing implementation for 64-bit atomic operations"
#  endif
#endif

/******************************************************************************/
/* 32-bit operations. */
#ifdef __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4
JEMALLOC_INLINE uint32_t
atomic_add_uint32(uint32_t *p, uint32_t x)
{

    return (__sync_add_and_fetch(p, x));
}

JEMALLOC_INLINE uint32_t
atomic_sub_uint32(uint32_t *p, uint32_t x)
{

    return (__sync_sub_and_fetch(p, x));
}
#elif (defined(_MSC_VER))
JEMALLOC_INLINE uint32_t
atomic_add_uint32(uint32_t *p, uint32_t x)
{

    return (InterlockedExchangeAdd(p, x));
}

JEMALLOC_INLINE uint32_t
atomic_sub_uint32(uint32_t *p, uint32_t x)
{

    return (InterlockedExchangeAdd(p, -((int32_t)x)));
}
#elif (defined(JEMALLOC_OSATOMIC))
JEMALLOC_INLINE uint32_t
atomic_add_uint32(uint32_t *p, uint32_t x)
{

    return (OSAtomicAdd32((int32_t)x, (int32_t *)p));
}

JEMALLOC_INLINE uint32_t
atomic_sub_uint32(uint32_t *p, uint32_t x)
{

    return (OSAtomicAdd32(-((int32_t)x), (int32_t *)p));
}
#elif (defined(__i386__) || defined(__amd64__) || defined(__x86_64__))
JEMALLOC_INLINE uint32_t
atomic_add_uint32(uint32_t *p, uint32_t x)
{

    asm volatile (
        "lock; xaddl %0, %1;"
        : "+r" (x), "=m" (*p) /* Outputs. */
        : "m" (*p) /* Inputs. */
        );

    return (x);
}

JEMALLOC_INLINE uint32_t
atomic_sub_uint32(uint32_t *p, uint32_t x)
{

    x = (uint32_t)(-(int32_t)x);
    asm volatile (
        "lock; xaddl %0, %1;"
        : "+r" (x), "=m" (*p) /* Outputs. */
        : "m" (*p) /* Inputs. */
        );

    return (x);
}
#elif (defined(JEMALLOC_ATOMIC9))
JEMALLOC_INLINE uint32_t
atomic_add_uint32(uint32_t *p, uint32_t x)
{

    return (atomic_fetchadd_32(p, x) + x);
}

JEMALLOC_INLINE uint32_t
atomic_sub_uint32(uint32_t *p, uint32_t x)
{

    return (atomic_fetchadd_32(p, (uint32_t)(-(int32_t)x)) - x);
}
#elif (defined(JE_FORCE_SYNC_COMPARE_AND_SWAP_4))
JEMALLOC_INLINE uint32_t
atomic_add_uint32(uint32_t *p, uint32_t x)
{

    return (__sync_add_and_fetch(p, x));
}

JEMALLOC_INLINE uint32_t
atomic_sub_uint32(uint32_t *p, uint32_t x)
{

    return (__sync_sub_and_fetch(p, x));
}
#else
#  error "Missing implementation for 32-bit atomic operations"
#endif

/******************************************************************************/
/* size_t operations. */
JEMALLOC_INLINE size_t
atomic_add_z(size_t *p, size_t x)
{

#if (LG_SIZEOF_PTR == 3)
    return ((size_t)atomic_add_uint64((uint64_t *)p, (uint64_t)x));
#elif (LG_SIZEOF_PTR == 2)
    return ((size_t)atomic_add_uint32((uint32_t *)p, (uint32_t)x));
#endif
}

JEMALLOC_INLINE size_t
atomic_sub_z(size_t *p, size_t x)
{

#if (LG_SIZEOF_PTR == 3)
    return ((size_t)atomic_add_uint64((uint64_t *)p,
        (uint64_t)-((int64_t)x)));
#elif (LG_SIZEOF_PTR == 2)
    return ((size_t)atomic_add_uint32((uint32_t *)p,
        (uint32_t)-((int32_t)x)));
#endif
}

/******************************************************************************/
/* unsigned operations. */
JEMALLOC_INLINE unsigned
atomic_add_u(unsigned *p, unsigned x)
{

#if (LG_SIZEOF_INT == 3)
    return ((unsigned)atomic_add_uint64((uint64_t *)p, (uint64_t)x));
#elif (LG_SIZEOF_INT == 2)
    return ((unsigned)atomic_add_uint32((uint32_t *)p, (uint32_t)x));
#endif
}

JEMALLOC_INLINE unsigned
atomic_sub_u(unsigned *p, unsigned x)
{

#if (LG_SIZEOF_INT == 3)
    return ((unsigned)atomic_add_uint64((uint64_t *)p,
        (uint64_t)-((int64_t)x)));
#elif (LG_SIZEOF_INT == 2)
    return ((unsigned)atomic_add_uint32((uint32_t *)p,
        (uint32_t)-((int32_t)x)));
#endif
}
/******************************************************************************/
#endif

#endif /* JEMALLOC_H_INLINES */
/******************************************************************************/
