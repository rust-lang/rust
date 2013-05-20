/******************************************************************************/
#ifdef JEMALLOC_H_TYPES

/* Size of stack-allocated buffer passed to buferror(). */
#define	BUFERROR_BUF		64

/*
 * Size of stack-allocated buffer used by malloc_{,v,vc}printf().  This must be
 * large enough for all possible uses within jemalloc.
 */
#define	MALLOC_PRINTF_BUFSIZE	4096

/*
 * Wrap a cpp argument that contains commas such that it isn't broken up into
 * multiple arguments.
 */
#define JEMALLOC_CONCAT(...) __VA_ARGS__

/*
 * Silence compiler warnings due to uninitialized values.  This is used
 * wherever the compiler fails to recognize that the variable is never used
 * uninitialized.
 */
#ifdef JEMALLOC_CC_SILENCE
#  define JEMALLOC_CC_SILENCE_INIT(v) = v
#else
#  define JEMALLOC_CC_SILENCE_INIT(v)
#endif

/*
 * Define a custom assert() in order to reduce the chances of deadlock during
 * assertion failure.
 */
#ifndef assert
#define	assert(e) do {							\
    if (config_debug && !(e)) {					\
        malloc_printf(						\
            "<jemalloc>: %s:%d: Failed assertion: \"%s\"\n",	\
            __FILE__, __LINE__, #e);				\
        abort();						\
    }								\
} while (0)
#endif

/* Use to assert a particular configuration, e.g., cassert(config_debug). */
#define	cassert(c) do {							\
    if ((c) == false)						\
        assert(false);						\
} while (0)

#ifndef not_reached
#define	not_reached() do {						\
    if (config_debug) {						\
        malloc_printf(						\
            "<jemalloc>: %s:%d: Unreachable code reached\n",	\
            __FILE__, __LINE__);				\
        abort();						\
    }								\
} while (0)
#endif

#ifndef not_implemented
#define	not_implemented() do {						\
    if (config_debug) {						\
        malloc_printf("<jemalloc>: %s:%d: Not implemented\n",	\
            __FILE__, __LINE__);				\
        abort();						\
    }								\
} while (0)
#endif

#define	assert_not_implemented(e) do {					\
    if (config_debug && !(e))					\
        not_implemented();					\
} while (0)

#endif /* JEMALLOC_H_TYPES */
/******************************************************************************/
#ifdef JEMALLOC_H_STRUCTS

#endif /* JEMALLOC_H_STRUCTS */
/******************************************************************************/
#ifdef JEMALLOC_H_EXTERNS

int	buferror(char *buf, size_t buflen);
uintmax_t	malloc_strtoumax(const char *nptr, char **endptr, int base);
void	malloc_write(const char *s);

/*
 * malloc_vsnprintf() supports a subset of snprintf(3) that avoids floating
 * point math.
 */
int	malloc_vsnprintf(char *str, size_t size, const char *format,
    va_list ap);
int	malloc_snprintf(char *str, size_t size, const char *format, ...)
    JEMALLOC_ATTR(format(printf, 3, 4));
void	malloc_vcprintf(void (*write_cb)(void *, const char *), void *cbopaque,
    const char *format, va_list ap);
void malloc_cprintf(void (*write)(void *, const char *), void *cbopaque,
    const char *format, ...) JEMALLOC_ATTR(format(printf, 3, 4));
void	malloc_printf(const char *format, ...)
    JEMALLOC_ATTR(format(printf, 1, 2));

#endif /* JEMALLOC_H_EXTERNS */
/******************************************************************************/
#ifdef JEMALLOC_H_INLINES

#ifndef JEMALLOC_ENABLE_INLINE
size_t	pow2_ceil(size_t x);
void	malloc_write(const char *s);
void	set_errno(int errnum);
int	get_errno(void);
#endif

#if (defined(JEMALLOC_ENABLE_INLINE) || defined(JEMALLOC_UTIL_C_))
/* Compute the smallest power of 2 that is >= x. */
JEMALLOC_INLINE size_t
pow2_ceil(size_t x)
{

    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
#if (LG_SIZEOF_PTR == 3)
    x |= x >> 32;
#endif
    x++;
    return (x);
}

/* Sets error code */
JEMALLOC_INLINE void
set_errno(int errnum)
{

#ifdef _WIN32
    SetLastError(errnum);
#else
    errno = errnum;
#endif
}

/* Get last error code */
JEMALLOC_INLINE int
get_errno(void)
{

#ifdef _WIN32
    return (GetLastError());
#else
    return (errno);
#endif
}
#endif

#endif /* JEMALLOC_H_INLINES */
/******************************************************************************/
