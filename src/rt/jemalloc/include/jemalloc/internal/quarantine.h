/******************************************************************************/
#ifdef JEMALLOC_H_TYPES

typedef struct quarantine_obj_s quarantine_obj_t;
typedef struct quarantine_s quarantine_t;

/* Default per thread quarantine size if valgrind is enabled. */
#define	JEMALLOC_VALGRIND_QUARANTINE_DEFAULT	(ZU(1) << 24)

#endif /* JEMALLOC_H_TYPES */
/******************************************************************************/
#ifdef JEMALLOC_H_STRUCTS

struct quarantine_obj_s {
    void	*ptr;
    size_t	usize;
};

struct quarantine_s {
    size_t			curbytes;
    size_t			curobjs;
    size_t			first;
#define	LG_MAXOBJS_INIT 10
    size_t			lg_maxobjs;
    quarantine_obj_t	objs[1]; /* Dynamically sized ring buffer. */
};

#endif /* JEMALLOC_H_STRUCTS */
/******************************************************************************/
#ifdef JEMALLOC_H_EXTERNS

quarantine_t	*quarantine_init(size_t lg_maxobjs);
void	quarantine(void *ptr);
void	quarantine_cleanup(void *arg);
bool	quarantine_boot(void);

#endif /* JEMALLOC_H_EXTERNS */
/******************************************************************************/
#ifdef JEMALLOC_H_INLINES

#ifndef JEMALLOC_ENABLE_INLINE
malloc_tsd_protos(JEMALLOC_ATTR(unused), quarantine, quarantine_t *)

void	quarantine_alloc_hook(void);
#endif

#if (defined(JEMALLOC_ENABLE_INLINE) || defined(JEMALLOC_QUARANTINE_C_))
malloc_tsd_externs(quarantine, quarantine_t *)
malloc_tsd_funcs(JEMALLOC_ALWAYS_INLINE, quarantine, quarantine_t *, NULL,
    quarantine_cleanup)

JEMALLOC_ALWAYS_INLINE void
quarantine_alloc_hook(void)
{
    quarantine_t *quarantine;

    assert(config_fill && opt_quarantine);

    quarantine = *quarantine_tsd_get();
    if (quarantine == NULL)
        quarantine_init(LG_MAXOBJS_INIT);
}
#endif

#endif /* JEMALLOC_H_INLINES */
/******************************************************************************/
