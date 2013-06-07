/******************************************************************************/
#ifdef JEMALLOC_H_TYPES

typedef struct prof_bt_s prof_bt_t;
typedef struct prof_cnt_s prof_cnt_t;
typedef struct prof_thr_cnt_s prof_thr_cnt_t;
typedef struct prof_ctx_s prof_ctx_t;
typedef struct prof_tdata_s prof_tdata_t;

/* Option defaults. */
#define	PROF_PREFIX_DEFAULT		"jeprof"
#define	LG_PROF_SAMPLE_DEFAULT		19
#define	LG_PROF_INTERVAL_DEFAULT	-1

/*
 * Hard limit on stack backtrace depth.  The version of prof_backtrace() that
 * is based on __builtin_return_address() necessarily has a hard-coded number
 * of backtrace frame handlers, and should be kept in sync with this setting.
 */
#define	PROF_BT_MAX			128

/* Maximum number of backtraces to store in each per thread LRU cache. */
#define	PROF_TCMAX			1024

/* Initial hash table size. */
#define	PROF_CKH_MINITEMS		64

/* Size of memory buffer to use when writing dump files. */
#define	PROF_DUMP_BUFSIZE		65536

/* Size of stack-allocated buffer used by prof_printf(). */
#define	PROF_PRINTF_BUFSIZE		128

/*
 * Number of mutexes shared among all ctx's.  No space is allocated for these
 * unless profiling is enabled, so it's okay to over-provision.
 */
#define	PROF_NCTX_LOCKS			1024

/*
 * prof_tdata pointers close to NULL are used to encode state information that
 * is used for cleaning up during thread shutdown.
 */
#define	PROF_TDATA_STATE_REINCARNATED	((prof_tdata_t *)(uintptr_t)1)
#define	PROF_TDATA_STATE_PURGATORY	((prof_tdata_t *)(uintptr_t)2)
#define	PROF_TDATA_STATE_MAX		PROF_TDATA_STATE_PURGATORY

#endif /* JEMALLOC_H_TYPES */
/******************************************************************************/
#ifdef JEMALLOC_H_STRUCTS

struct prof_bt_s {
    /* Backtrace, stored as len program counters. */
    void		**vec;
    unsigned	len;
};

#ifdef JEMALLOC_PROF_LIBGCC
/* Data structure passed to libgcc _Unwind_Backtrace() callback functions. */
typedef struct {
    prof_bt_t	*bt;
    unsigned	nignore;
    unsigned	max;
} prof_unwind_data_t;
#endif

struct prof_cnt_s {
    /*
     * Profiling counters.  An allocation/deallocation pair can operate on
     * different prof_thr_cnt_t objects that are linked into the same
     * prof_ctx_t cnts_ql, so it is possible for the cur* counters to go
     * negative.  In principle it is possible for the *bytes counters to
     * overflow/underflow, but a general solution would require something
     * like 128-bit counters; this implementation doesn't bother to solve
     * that problem.
     */
    int64_t		curobjs;
    int64_t		curbytes;
    uint64_t	accumobjs;
    uint64_t	accumbytes;
};

struct prof_thr_cnt_s {
    /* Linkage into prof_ctx_t's cnts_ql. */
    ql_elm(prof_thr_cnt_t)	cnts_link;

    /* Linkage into thread's LRU. */
    ql_elm(prof_thr_cnt_t)	lru_link;

    /*
     * Associated context.  If a thread frees an object that it did not
     * allocate, it is possible that the context is not cached in the
     * thread's hash table, in which case it must be able to look up the
     * context, insert a new prof_thr_cnt_t into the thread's hash table,
     * and link it into the prof_ctx_t's cnts_ql.
     */
    prof_ctx_t		*ctx;

    /*
     * Threads use memory barriers to update the counters.  Since there is
     * only ever one writer, the only challenge is for the reader to get a
     * consistent read of the counters.
     *
     * The writer uses this series of operations:
     *
     * 1) Increment epoch to an odd number.
     * 2) Update counters.
     * 3) Increment epoch to an even number.
     *
     * The reader must assure 1) that the epoch is even while it reads the
     * counters, and 2) that the epoch doesn't change between the time it
     * starts and finishes reading the counters.
     */
    unsigned		epoch;

    /* Profiling counters. */
    prof_cnt_t		cnts;
};

struct prof_ctx_s {
    /* Associated backtrace. */
    prof_bt_t		*bt;

    /* Protects nlimbo, cnt_merged, and cnts_ql. */
    malloc_mutex_t		*lock;

    /*
     * Number of threads that currently cause this ctx to be in a state of
     * limbo due to one of:
     *   - Initializing per thread counters associated with this ctx.
     *   - Preparing to destroy this ctx.
     * nlimbo must be 1 (single destroyer) in order to safely destroy the
     * ctx.
     */
    unsigned		nlimbo;

    /* Temporary storage for summation during dump. */
    prof_cnt_t		cnt_summed;

    /* When threads exit, they merge their stats into cnt_merged. */
    prof_cnt_t		cnt_merged;

    /*
     * List of profile counters, one for each thread that has allocated in
     * this context.
     */
    ql_head(prof_thr_cnt_t)	cnts_ql;
};

struct prof_tdata_s {
    /*
     * Hash of (prof_bt_t *)-->(prof_thr_cnt_t *).  Each thread keeps a
     * cache of backtraces, with associated thread-specific prof_thr_cnt_t
     * objects.  Other threads may read the prof_thr_cnt_t contents, but no
     * others will ever write them.
     *
     * Upon thread exit, the thread must merge all the prof_thr_cnt_t
     * counter data into the associated prof_ctx_t objects, and unlink/free
     * the prof_thr_cnt_t objects.
     */
    ckh_t			bt2cnt;

    /* LRU for contents of bt2cnt. */
    ql_head(prof_thr_cnt_t)	lru_ql;

    /* Backtrace vector, used for calls to prof_backtrace(). */
    void			**vec;

    /* Sampling state. */
    uint64_t		prng_state;
    uint64_t		threshold;
    uint64_t		accum;

    /* State used to avoid dumping while operating on prof internals. */
    bool			enq;
    bool			enq_idump;
    bool			enq_gdump;
};

#endif /* JEMALLOC_H_STRUCTS */
/******************************************************************************/
#ifdef JEMALLOC_H_EXTERNS

extern bool	opt_prof;
/*
 * Even if opt_prof is true, sampling can be temporarily disabled by setting
 * opt_prof_active to false.  No locking is used when updating opt_prof_active,
 * so there are no guarantees regarding how long it will take for all threads
 * to notice state changes.
 */
extern bool	opt_prof_active;
extern size_t	opt_lg_prof_sample;   /* Mean bytes between samples. */
extern ssize_t	opt_lg_prof_interval; /* lg(prof_interval). */
extern bool	opt_prof_gdump;       /* High-water memory dumping. */
extern bool	opt_prof_final;       /* Final profile dumping. */
extern bool	opt_prof_leak;        /* Dump leak summary at exit. */
extern bool	opt_prof_accum;       /* Report cumulative bytes. */
extern char	opt_prof_prefix[PATH_MAX + 1];

/*
 * Profile dump interval, measured in bytes allocated.  Each arena triggers a
 * profile dump when it reaches this threshold.  The effect is that the
 * interval between profile dumps averages prof_interval, though the actual
 * interval between dumps will tend to be sporadic, and the interval will be a
 * maximum of approximately (prof_interval * narenas).
 */
extern uint64_t	prof_interval;

/*
 * If true, promote small sampled objects to large objects, since small run
 * headers do not have embedded profile context pointers.
 */
extern bool	prof_promote;

void	bt_init(prof_bt_t *bt, void **vec);
void	prof_backtrace(prof_bt_t *bt, unsigned nignore);
prof_thr_cnt_t	*prof_lookup(prof_bt_t *bt);
void	prof_idump(void);
bool	prof_mdump(const char *filename);
void	prof_gdump(void);
prof_tdata_t	*prof_tdata_init(void);
void	prof_tdata_cleanup(void *arg);
void	prof_boot0(void);
void	prof_boot1(void);
bool	prof_boot2(void);
void	prof_prefork(void);
void	prof_postfork_parent(void);
void	prof_postfork_child(void);

#endif /* JEMALLOC_H_EXTERNS */
/******************************************************************************/
#ifdef JEMALLOC_H_INLINES

#define	PROF_ALLOC_PREP(nignore, size, ret) do {			\
    prof_tdata_t *prof_tdata;					\
    prof_bt_t bt;							\
                                    \
    assert(size == s2u(size));					\
                                    \
    prof_tdata = prof_tdata_get(true);				\
    if ((uintptr_t)prof_tdata <= (uintptr_t)PROF_TDATA_STATE_MAX) {	\
        if (prof_tdata != NULL)					\
            ret = (prof_thr_cnt_t *)(uintptr_t)1U;		\
        else							\
            ret = NULL;					\
        break;							\
    }								\
                                    \
    if (opt_prof_active == false) {					\
        /* Sampling is currently inactive, so avoid sampling. */\
        ret = (prof_thr_cnt_t *)(uintptr_t)1U;			\
    } else if (opt_lg_prof_sample == 0) {				\
        /* Don't bother with sampling logic, since sampling   */\
        /* interval is 1.                                     */\
        bt_init(&bt, prof_tdata->vec);				\
        prof_backtrace(&bt, nignore);				\
        ret = prof_lookup(&bt);					\
    } else {							\
        if (prof_tdata->threshold == 0) {			\
            /* Initialize.  Seed the prng differently for */\
            /* each thread.                               */\
            prof_tdata->prng_state =			\
                (uint64_t)(uintptr_t)&size;			\
            prof_sample_threshold_update(prof_tdata);	\
        }							\
                                    \
        /* Determine whether to capture a backtrace based on  */\
        /* whether size is enough for prof_accum to reach     */\
        /* prof_tdata->threshold.  However, delay updating    */\
        /* these variables until prof_{m,re}alloc(), because  */\
        /* we don't know for sure that the allocation will    */\
        /* succeed.                                           */\
        /*                                                    */\
        /* Use subtraction rather than addition to avoid      */\
        /* potential integer overflow.                        */\
        if (size >= prof_tdata->threshold -			\
            prof_tdata->accum) {				\
            bt_init(&bt, prof_tdata->vec);			\
            prof_backtrace(&bt, nignore);			\
            ret = prof_lookup(&bt);				\
        } else							\
            ret = (prof_thr_cnt_t *)(uintptr_t)1U;		\
    }								\
} while (0)

#ifndef JEMALLOC_ENABLE_INLINE
malloc_tsd_protos(JEMALLOC_ATTR(unused), prof_tdata, prof_tdata_t *)

prof_tdata_t	*prof_tdata_get(bool create);
void	prof_sample_threshold_update(prof_tdata_t *prof_tdata);
prof_ctx_t	*prof_ctx_get(const void *ptr);
void	prof_ctx_set(const void *ptr, prof_ctx_t *ctx);
bool	prof_sample_accum_update(size_t size);
void	prof_malloc(const void *ptr, size_t size, prof_thr_cnt_t *cnt);
void	prof_realloc(const void *ptr, size_t size, prof_thr_cnt_t *cnt,
    size_t old_size, prof_ctx_t *old_ctx);
void	prof_free(const void *ptr, size_t size);
#endif

#if (defined(JEMALLOC_ENABLE_INLINE) || defined(JEMALLOC_PROF_C_))
/* Thread-specific backtrace cache, used to reduce bt2ctx contention. */
malloc_tsd_externs(prof_tdata, prof_tdata_t *)
malloc_tsd_funcs(JEMALLOC_INLINE, prof_tdata, prof_tdata_t *, NULL,
    prof_tdata_cleanup)

JEMALLOC_INLINE prof_tdata_t *
prof_tdata_get(bool create)
{
    prof_tdata_t *prof_tdata;

    cassert(config_prof);

    prof_tdata = *prof_tdata_tsd_get();
    if (create && prof_tdata == NULL)
        prof_tdata = prof_tdata_init();

    return (prof_tdata);
}

JEMALLOC_INLINE void
prof_sample_threshold_update(prof_tdata_t *prof_tdata)
{
    uint64_t r;
    double u;

    cassert(config_prof);

    /*
     * Compute sample threshold as a geometrically distributed random
     * variable with mean (2^opt_lg_prof_sample).
     *
     *                         __        __
     *                         |  log(u)  |                     1
     * prof_tdata->threshold = | -------- |, where p = -------------------
     *                         | log(1-p) |             opt_lg_prof_sample
     *                                                 2
     *
     * For more information on the math, see:
     *
     *   Non-Uniform Random Variate Generation
     *   Luc Devroye
     *   Springer-Verlag, New York, 1986
     *   pp 500
     *   (http://cg.scs.carleton.ca/~luc/rnbookindex.html)
     */
    prng64(r, 53, prof_tdata->prng_state,
        UINT64_C(6364136223846793005), UINT64_C(1442695040888963407));
    u = (double)r * (1.0/9007199254740992.0L);
    prof_tdata->threshold = (uint64_t)(log(u) /
        log(1.0 - (1.0 / (double)((uint64_t)1U << opt_lg_prof_sample))))
        + (uint64_t)1U;
}

JEMALLOC_INLINE prof_ctx_t *
prof_ctx_get(const void *ptr)
{
    prof_ctx_t *ret;
    arena_chunk_t *chunk;

    cassert(config_prof);
    assert(ptr != NULL);

    chunk = (arena_chunk_t *)CHUNK_ADDR2BASE(ptr);
    if (chunk != ptr) {
        /* Region. */
        ret = arena_prof_ctx_get(ptr);
    } else
        ret = huge_prof_ctx_get(ptr);

    return (ret);
}

JEMALLOC_INLINE void
prof_ctx_set(const void *ptr, prof_ctx_t *ctx)
{
    arena_chunk_t *chunk;

    cassert(config_prof);
    assert(ptr != NULL);

    chunk = (arena_chunk_t *)CHUNK_ADDR2BASE(ptr);
    if (chunk != ptr) {
        /* Region. */
        arena_prof_ctx_set(ptr, ctx);
    } else
        huge_prof_ctx_set(ptr, ctx);
}

JEMALLOC_INLINE bool
prof_sample_accum_update(size_t size)
{
    prof_tdata_t *prof_tdata;

    cassert(config_prof);
    /* Sampling logic is unnecessary if the interval is 1. */
    assert(opt_lg_prof_sample != 0);

    prof_tdata = prof_tdata_get(false);
    if ((uintptr_t)prof_tdata <= (uintptr_t)PROF_TDATA_STATE_MAX)
        return (true);

    /* Take care to avoid integer overflow. */
    if (size >= prof_tdata->threshold - prof_tdata->accum) {
        prof_tdata->accum -= (prof_tdata->threshold - size);
        /* Compute new sample threshold. */
        prof_sample_threshold_update(prof_tdata);
        while (prof_tdata->accum >= prof_tdata->threshold) {
            prof_tdata->accum -= prof_tdata->threshold;
            prof_sample_threshold_update(prof_tdata);
        }
        return (false);
    } else {
        prof_tdata->accum += size;
        return (true);
    }
}

JEMALLOC_INLINE void
prof_malloc(const void *ptr, size_t size, prof_thr_cnt_t *cnt)
{

    cassert(config_prof);
    assert(ptr != NULL);
    assert(size == isalloc(ptr, true));

    if (opt_lg_prof_sample != 0) {
        if (prof_sample_accum_update(size)) {
            /*
             * Don't sample.  For malloc()-like allocation, it is
             * always possible to tell in advance how large an
             * object's usable size will be, so there should never
             * be a difference between the size passed to
             * PROF_ALLOC_PREP() and prof_malloc().
             */
            assert((uintptr_t)cnt == (uintptr_t)1U);
        }
    }

    if ((uintptr_t)cnt > (uintptr_t)1U) {
        prof_ctx_set(ptr, cnt->ctx);

        cnt->epoch++;
        /*********/
        mb_write();
        /*********/
        cnt->cnts.curobjs++;
        cnt->cnts.curbytes += size;
        if (opt_prof_accum) {
            cnt->cnts.accumobjs++;
            cnt->cnts.accumbytes += size;
        }
        /*********/
        mb_write();
        /*********/
        cnt->epoch++;
        /*********/
        mb_write();
        /*********/
    } else
        prof_ctx_set(ptr, (prof_ctx_t *)(uintptr_t)1U);
}

JEMALLOC_INLINE void
prof_realloc(const void *ptr, size_t size, prof_thr_cnt_t *cnt,
    size_t old_size, prof_ctx_t *old_ctx)
{
    prof_thr_cnt_t *told_cnt;

    cassert(config_prof);
    assert(ptr != NULL || (uintptr_t)cnt <= (uintptr_t)1U);

    if (ptr != NULL) {
        assert(size == isalloc(ptr, true));
        if (opt_lg_prof_sample != 0) {
            if (prof_sample_accum_update(size)) {
                /*
                 * Don't sample.  The size passed to
                 * PROF_ALLOC_PREP() was larger than what
                 * actually got allocated, so a backtrace was
                 * captured for this allocation, even though
                 * its actual size was insufficient to cross
                 * the sample threshold.
                 */
                cnt = (prof_thr_cnt_t *)(uintptr_t)1U;
            }
        }
    }

    if ((uintptr_t)old_ctx > (uintptr_t)1U) {
        told_cnt = prof_lookup(old_ctx->bt);
        if (told_cnt == NULL) {
            /*
             * It's too late to propagate OOM for this realloc(),
             * so operate directly on old_cnt->ctx->cnt_merged.
             */
            malloc_mutex_lock(old_ctx->lock);
            old_ctx->cnt_merged.curobjs--;
            old_ctx->cnt_merged.curbytes -= old_size;
            malloc_mutex_unlock(old_ctx->lock);
            told_cnt = (prof_thr_cnt_t *)(uintptr_t)1U;
        }
    } else
        told_cnt = (prof_thr_cnt_t *)(uintptr_t)1U;

    if ((uintptr_t)told_cnt > (uintptr_t)1U)
        told_cnt->epoch++;
    if ((uintptr_t)cnt > (uintptr_t)1U) {
        prof_ctx_set(ptr, cnt->ctx);
        cnt->epoch++;
    } else if (ptr != NULL)
        prof_ctx_set(ptr, (prof_ctx_t *)(uintptr_t)1U);
    /*********/
    mb_write();
    /*********/
    if ((uintptr_t)told_cnt > (uintptr_t)1U) {
        told_cnt->cnts.curobjs--;
        told_cnt->cnts.curbytes -= old_size;
    }
    if ((uintptr_t)cnt > (uintptr_t)1U) {
        cnt->cnts.curobjs++;
        cnt->cnts.curbytes += size;
        if (opt_prof_accum) {
            cnt->cnts.accumobjs++;
            cnt->cnts.accumbytes += size;
        }
    }
    /*********/
    mb_write();
    /*********/
    if ((uintptr_t)told_cnt > (uintptr_t)1U)
        told_cnt->epoch++;
    if ((uintptr_t)cnt > (uintptr_t)1U)
        cnt->epoch++;
    /*********/
    mb_write(); /* Not strictly necessary. */
}

JEMALLOC_INLINE void
prof_free(const void *ptr, size_t size)
{
    prof_ctx_t *ctx = prof_ctx_get(ptr);

    cassert(config_prof);

    if ((uintptr_t)ctx > (uintptr_t)1) {
        prof_thr_cnt_t *tcnt;
        assert(size == isalloc(ptr, true));
        tcnt = prof_lookup(ctx->bt);

        if (tcnt != NULL) {
            tcnt->epoch++;
            /*********/
            mb_write();
            /*********/
            tcnt->cnts.curobjs--;
            tcnt->cnts.curbytes -= size;
            /*********/
            mb_write();
            /*********/
            tcnt->epoch++;
            /*********/
            mb_write();
            /*********/
        } else {
            /*
             * OOM during free() cannot be propagated, so operate
             * directly on cnt->ctx->cnt_merged.
             */
            malloc_mutex_lock(ctx->lock);
            ctx->cnt_merged.curobjs--;
            ctx->cnt_merged.curbytes -= size;
            malloc_mutex_unlock(ctx->lock);
        }
    }
}
#endif

#endif /* JEMALLOC_H_INLINES */
/******************************************************************************/
