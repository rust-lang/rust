#define	JEMALLOC_PROF_C_
#include "jemalloc/internal/jemalloc_internal.h"
/******************************************************************************/

#ifdef JEMALLOC_PROF_LIBUNWIND
#define	UNW_LOCAL_ONLY
#include <libunwind.h>
#endif

#ifdef JEMALLOC_PROF_LIBGCC
#include <unwind.h>
#endif

/******************************************************************************/
/* Data. */

malloc_tsd_data(, prof_tdata, prof_tdata_t *, NULL)

bool		opt_prof = false;
bool		opt_prof_active = true;
size_t		opt_lg_prof_sample = LG_PROF_SAMPLE_DEFAULT;
ssize_t		opt_lg_prof_interval = LG_PROF_INTERVAL_DEFAULT;
bool		opt_prof_gdump = false;
bool		opt_prof_final = true;
bool		opt_prof_leak = false;
bool		opt_prof_accum = false;
char		opt_prof_prefix[PATH_MAX + 1];

uint64_t	prof_interval = 0;
bool		prof_promote;

/*
 * Table of mutexes that are shared among ctx's.  These are leaf locks, so
 * there is no problem with using them for more than one ctx at the same time.
 * The primary motivation for this sharing though is that ctx's are ephemeral,
 * and destroying mutexes causes complications for systems that allocate when
 * creating/destroying mutexes.
 */
static malloc_mutex_t	*ctx_locks;
static unsigned		cum_ctxs; /* Atomic counter. */

/*
 * Global hash of (prof_bt_t *)-->(prof_ctx_t *).  This is the master data
 * structure that knows about all backtraces currently captured.
 */
static ckh_t		bt2ctx;
static malloc_mutex_t	bt2ctx_mtx;

static malloc_mutex_t	prof_dump_seq_mtx;
static uint64_t		prof_dump_seq;
static uint64_t		prof_dump_iseq;
static uint64_t		prof_dump_mseq;
static uint64_t		prof_dump_useq;

/*
 * This buffer is rather large for stack allocation, so use a single buffer for
 * all profile dumps.  The buffer is implicitly protected by bt2ctx_mtx, since
 * it must be locked anyway during dumping.
 */
static char		prof_dump_buf[PROF_DUMP_BUFSIZE];
static unsigned		prof_dump_buf_end;
static int		prof_dump_fd;

/* Do not dump any profiles until bootstrapping is complete. */
static bool		prof_booted = false;

/******************************************************************************/
/* Function prototypes for non-inline static functions. */

static prof_bt_t	*bt_dup(prof_bt_t *bt);
static void	bt_destroy(prof_bt_t *bt);
#ifdef JEMALLOC_PROF_LIBGCC
static _Unwind_Reason_Code	prof_unwind_init_callback(
    struct _Unwind_Context *context, void *arg);
static _Unwind_Reason_Code	prof_unwind_callback(
    struct _Unwind_Context *context, void *arg);
#endif
static bool	prof_flush(bool propagate_err);
static bool	prof_write(bool propagate_err, const char *s);
static bool	prof_printf(bool propagate_err, const char *format, ...)
    JEMALLOC_ATTR(format(printf, 2, 3));
static void	prof_ctx_sum(prof_ctx_t *ctx, prof_cnt_t *cnt_all,
    size_t *leak_nctx);
static void	prof_ctx_destroy(prof_ctx_t *ctx);
static void	prof_ctx_merge(prof_ctx_t *ctx, prof_thr_cnt_t *cnt);
static bool	prof_dump_ctx(bool propagate_err, prof_ctx_t *ctx,
    prof_bt_t *bt);
static bool	prof_dump_maps(bool propagate_err);
static bool	prof_dump(bool propagate_err, const char *filename,
    bool leakcheck);
static void	prof_dump_filename(char *filename, char v, int64_t vseq);
static void	prof_fdump(void);
static void	prof_bt_hash(const void *key, size_t r_hash[2]);
static bool	prof_bt_keycomp(const void *k1, const void *k2);
static malloc_mutex_t	*prof_ctx_mutex_choose(void);

/******************************************************************************/

void
bt_init(prof_bt_t *bt, void **vec)
{

	cassert(config_prof);

	bt->vec = vec;
	bt->len = 0;
}

static void
bt_destroy(prof_bt_t *bt)
{

	cassert(config_prof);

	idalloc(bt);
}

static prof_bt_t *
bt_dup(prof_bt_t *bt)
{
	prof_bt_t *ret;

	cassert(config_prof);

	/*
	 * Create a single allocation that has space for vec immediately
	 * following the prof_bt_t structure.  The backtraces that get
	 * stored in the backtrace caches are copied from stack-allocated
	 * temporary variables, so size is known at creation time.  Making this
	 * a contiguous object improves cache locality.
	 */
	ret = (prof_bt_t *)imalloc(QUANTUM_CEILING(sizeof(prof_bt_t)) +
	    (bt->len * sizeof(void *)));
	if (ret == NULL)
		return (NULL);
	ret->vec = (void **)((uintptr_t)ret +
	    QUANTUM_CEILING(sizeof(prof_bt_t)));
	memcpy(ret->vec, bt->vec, bt->len * sizeof(void *));
	ret->len = bt->len;

	return (ret);
}

static inline void
prof_enter(prof_tdata_t *prof_tdata)
{

	cassert(config_prof);

	assert(prof_tdata->enq == false);
	prof_tdata->enq = true;

	malloc_mutex_lock(&bt2ctx_mtx);
}

static inline void
prof_leave(prof_tdata_t *prof_tdata)
{
	bool idump, gdump;

	cassert(config_prof);

	malloc_mutex_unlock(&bt2ctx_mtx);

	assert(prof_tdata->enq);
	prof_tdata->enq = false;
	idump = prof_tdata->enq_idump;
	prof_tdata->enq_idump = false;
	gdump = prof_tdata->enq_gdump;
	prof_tdata->enq_gdump = false;

	if (idump)
		prof_idump();
	if (gdump)
		prof_gdump();
}

#ifdef JEMALLOC_PROF_LIBUNWIND
void
prof_backtrace(prof_bt_t *bt, unsigned nignore)
{
	unw_context_t uc;
	unw_cursor_t cursor;
	unsigned i;
	int err;

	cassert(config_prof);
	assert(bt->len == 0);
	assert(bt->vec != NULL);

	unw_getcontext(&uc);
	unw_init_local(&cursor, &uc);

	/* Throw away (nignore+1) stack frames, if that many exist. */
	for (i = 0; i < nignore + 1; i++) {
		err = unw_step(&cursor);
		if (err <= 0)
			return;
	}

	/*
	 * Iterate over stack frames until there are no more, or until no space
	 * remains in bt.
	 */
	for (i = 0; i < PROF_BT_MAX; i++) {
		unw_get_reg(&cursor, UNW_REG_IP, (unw_word_t *)&bt->vec[i]);
		bt->len++;
		err = unw_step(&cursor);
		if (err <= 0)
			break;
	}
}
#elif (defined(JEMALLOC_PROF_LIBGCC))
static _Unwind_Reason_Code
prof_unwind_init_callback(struct _Unwind_Context *context, void *arg)
{

	cassert(config_prof);

	return (_URC_NO_REASON);
}

static _Unwind_Reason_Code
prof_unwind_callback(struct _Unwind_Context *context, void *arg)
{
	prof_unwind_data_t *data = (prof_unwind_data_t *)arg;

	cassert(config_prof);

	if (data->nignore > 0)
		data->nignore--;
	else {
		data->bt->vec[data->bt->len] = (void *)_Unwind_GetIP(context);
		data->bt->len++;
		if (data->bt->len == data->max)
			return (_URC_END_OF_STACK);
	}

	return (_URC_NO_REASON);
}

void
prof_backtrace(prof_bt_t *bt, unsigned nignore)
{
	prof_unwind_data_t data = {bt, nignore, PROF_BT_MAX};

	cassert(config_prof);

	_Unwind_Backtrace(prof_unwind_callback, &data);
}
#elif (defined(JEMALLOC_PROF_GCC))
void
prof_backtrace(prof_bt_t *bt, unsigned nignore)
{
#define	BT_FRAME(i)							\
	if ((i) < nignore + PROF_BT_MAX) {				\
		void *p;						\
		if (__builtin_frame_address(i) == 0)			\
			return;						\
		p = __builtin_return_address(i);			\
		if (p == NULL)						\
			return;						\
		if (i >= nignore) {					\
			bt->vec[(i) - nignore] = p;			\
			bt->len = (i) - nignore + 1;			\
		}							\
	} else								\
		return;

	cassert(config_prof);
	assert(nignore <= 3);

	BT_FRAME(0)
	BT_FRAME(1)
	BT_FRAME(2)
	BT_FRAME(3)
	BT_FRAME(4)
	BT_FRAME(5)
	BT_FRAME(6)
	BT_FRAME(7)
	BT_FRAME(8)
	BT_FRAME(9)

	BT_FRAME(10)
	BT_FRAME(11)
	BT_FRAME(12)
	BT_FRAME(13)
	BT_FRAME(14)
	BT_FRAME(15)
	BT_FRAME(16)
	BT_FRAME(17)
	BT_FRAME(18)
	BT_FRAME(19)

	BT_FRAME(20)
	BT_FRAME(21)
	BT_FRAME(22)
	BT_FRAME(23)
	BT_FRAME(24)
	BT_FRAME(25)
	BT_FRAME(26)
	BT_FRAME(27)
	BT_FRAME(28)
	BT_FRAME(29)

	BT_FRAME(30)
	BT_FRAME(31)
	BT_FRAME(32)
	BT_FRAME(33)
	BT_FRAME(34)
	BT_FRAME(35)
	BT_FRAME(36)
	BT_FRAME(37)
	BT_FRAME(38)
	BT_FRAME(39)

	BT_FRAME(40)
	BT_FRAME(41)
	BT_FRAME(42)
	BT_FRAME(43)
	BT_FRAME(44)
	BT_FRAME(45)
	BT_FRAME(46)
	BT_FRAME(47)
	BT_FRAME(48)
	BT_FRAME(49)

	BT_FRAME(50)
	BT_FRAME(51)
	BT_FRAME(52)
	BT_FRAME(53)
	BT_FRAME(54)
	BT_FRAME(55)
	BT_FRAME(56)
	BT_FRAME(57)
	BT_FRAME(58)
	BT_FRAME(59)

	BT_FRAME(60)
	BT_FRAME(61)
	BT_FRAME(62)
	BT_FRAME(63)
	BT_FRAME(64)
	BT_FRAME(65)
	BT_FRAME(66)
	BT_FRAME(67)
	BT_FRAME(68)
	BT_FRAME(69)

	BT_FRAME(70)
	BT_FRAME(71)
	BT_FRAME(72)
	BT_FRAME(73)
	BT_FRAME(74)
	BT_FRAME(75)
	BT_FRAME(76)
	BT_FRAME(77)
	BT_FRAME(78)
	BT_FRAME(79)

	BT_FRAME(80)
	BT_FRAME(81)
	BT_FRAME(82)
	BT_FRAME(83)
	BT_FRAME(84)
	BT_FRAME(85)
	BT_FRAME(86)
	BT_FRAME(87)
	BT_FRAME(88)
	BT_FRAME(89)

	BT_FRAME(90)
	BT_FRAME(91)
	BT_FRAME(92)
	BT_FRAME(93)
	BT_FRAME(94)
	BT_FRAME(95)
	BT_FRAME(96)
	BT_FRAME(97)
	BT_FRAME(98)
	BT_FRAME(99)

	BT_FRAME(100)
	BT_FRAME(101)
	BT_FRAME(102)
	BT_FRAME(103)
	BT_FRAME(104)
	BT_FRAME(105)
	BT_FRAME(106)
	BT_FRAME(107)
	BT_FRAME(108)
	BT_FRAME(109)

	BT_FRAME(110)
	BT_FRAME(111)
	BT_FRAME(112)
	BT_FRAME(113)
	BT_FRAME(114)
	BT_FRAME(115)
	BT_FRAME(116)
	BT_FRAME(117)
	BT_FRAME(118)
	BT_FRAME(119)

	BT_FRAME(120)
	BT_FRAME(121)
	BT_FRAME(122)
	BT_FRAME(123)
	BT_FRAME(124)
	BT_FRAME(125)
	BT_FRAME(126)
	BT_FRAME(127)

	/* Extras to compensate for nignore. */
	BT_FRAME(128)
	BT_FRAME(129)
	BT_FRAME(130)
#undef BT_FRAME
}
#else
void
prof_backtrace(prof_bt_t *bt, unsigned nignore)
{

	cassert(config_prof);
	assert(false);
}
#endif

prof_thr_cnt_t *
prof_lookup(prof_bt_t *bt)
{
	union {
		prof_thr_cnt_t	*p;
		void		*v;
	} ret;
	prof_tdata_t *prof_tdata;

	cassert(config_prof);

	prof_tdata = prof_tdata_get(false);
	if ((uintptr_t)prof_tdata <= (uintptr_t)PROF_TDATA_STATE_MAX)
		return (NULL);

	if (ckh_search(&prof_tdata->bt2cnt, bt, NULL, &ret.v)) {
		union {
			prof_bt_t	*p;
			void		*v;
		} btkey;
		union {
			prof_ctx_t	*p;
			void		*v;
		} ctx;
		bool new_ctx;

		/*
		 * This thread's cache lacks bt.  Look for it in the global
		 * cache.
		 */
		prof_enter(prof_tdata);
		if (ckh_search(&bt2ctx, bt, &btkey.v, &ctx.v)) {
			/* bt has never been seen before.  Insert it. */
			ctx.v = imalloc(sizeof(prof_ctx_t));
			if (ctx.v == NULL) {
				prof_leave(prof_tdata);
				return (NULL);
			}
			btkey.p = bt_dup(bt);
			if (btkey.v == NULL) {
				prof_leave(prof_tdata);
				idalloc(ctx.v);
				return (NULL);
			}
			ctx.p->bt = btkey.p;
			ctx.p->lock = prof_ctx_mutex_choose();
			/*
			 * Set nlimbo to 1, in order to avoid a race condition
			 * with prof_ctx_merge()/prof_ctx_destroy().
			 */
			ctx.p->nlimbo = 1;
			memset(&ctx.p->cnt_merged, 0, sizeof(prof_cnt_t));
			ql_new(&ctx.p->cnts_ql);
			if (ckh_insert(&bt2ctx, btkey.v, ctx.v)) {
				/* OOM. */
				prof_leave(prof_tdata);
				idalloc(btkey.v);
				idalloc(ctx.v);
				return (NULL);
			}
			new_ctx = true;
		} else {
			/*
			 * Increment nlimbo, in order to avoid a race condition
			 * with prof_ctx_merge()/prof_ctx_destroy().
			 */
			malloc_mutex_lock(ctx.p->lock);
			ctx.p->nlimbo++;
			malloc_mutex_unlock(ctx.p->lock);
			new_ctx = false;
		}
		prof_leave(prof_tdata);

		/* Link a prof_thd_cnt_t into ctx for this thread. */
		if (ckh_count(&prof_tdata->bt2cnt) == PROF_TCMAX) {
			assert(ckh_count(&prof_tdata->bt2cnt) > 0);
			/*
			 * Flush the least recently used cnt in order to keep
			 * bt2cnt from becoming too large.
			 */
			ret.p = ql_last(&prof_tdata->lru_ql, lru_link);
			assert(ret.v != NULL);
			if (ckh_remove(&prof_tdata->bt2cnt, ret.p->ctx->bt,
			    NULL, NULL))
				assert(false);
			ql_remove(&prof_tdata->lru_ql, ret.p, lru_link);
			prof_ctx_merge(ret.p->ctx, ret.p);
			/* ret can now be re-used. */
		} else {
			assert(ckh_count(&prof_tdata->bt2cnt) < PROF_TCMAX);
			/* Allocate and partially initialize a new cnt. */
			ret.v = imalloc(sizeof(prof_thr_cnt_t));
			if (ret.p == NULL) {
				if (new_ctx)
					prof_ctx_destroy(ctx.p);
				return (NULL);
			}
			ql_elm_new(ret.p, cnts_link);
			ql_elm_new(ret.p, lru_link);
		}
		/* Finish initializing ret. */
		ret.p->ctx = ctx.p;
		ret.p->epoch = 0;
		memset(&ret.p->cnts, 0, sizeof(prof_cnt_t));
		if (ckh_insert(&prof_tdata->bt2cnt, btkey.v, ret.v)) {
			if (new_ctx)
				prof_ctx_destroy(ctx.p);
			idalloc(ret.v);
			return (NULL);
		}
		ql_head_insert(&prof_tdata->lru_ql, ret.p, lru_link);
		malloc_mutex_lock(ctx.p->lock);
		ql_tail_insert(&ctx.p->cnts_ql, ret.p, cnts_link);
		ctx.p->nlimbo--;
		malloc_mutex_unlock(ctx.p->lock);
	} else {
		/* Move ret to the front of the LRU. */
		ql_remove(&prof_tdata->lru_ql, ret.p, lru_link);
		ql_head_insert(&prof_tdata->lru_ql, ret.p, lru_link);
	}

	return (ret.p);
}

static bool
prof_flush(bool propagate_err)
{
	bool ret = false;
	ssize_t err;

	cassert(config_prof);

	err = write(prof_dump_fd, prof_dump_buf, prof_dump_buf_end);
	if (err == -1) {
		if (propagate_err == false) {
			malloc_write("<jemalloc>: write() failed during heap "
			    "profile flush\n");
			if (opt_abort)
				abort();
		}
		ret = true;
	}
	prof_dump_buf_end = 0;

	return (ret);
}

static bool
prof_write(bool propagate_err, const char *s)
{
	unsigned i, slen, n;

	cassert(config_prof);

	i = 0;
	slen = strlen(s);
	while (i < slen) {
		/* Flush the buffer if it is full. */
		if (prof_dump_buf_end == PROF_DUMP_BUFSIZE)
			if (prof_flush(propagate_err) && propagate_err)
				return (true);

		if (prof_dump_buf_end + slen <= PROF_DUMP_BUFSIZE) {
			/* Finish writing. */
			n = slen - i;
		} else {
			/* Write as much of s as will fit. */
			n = PROF_DUMP_BUFSIZE - prof_dump_buf_end;
		}
		memcpy(&prof_dump_buf[prof_dump_buf_end], &s[i], n);
		prof_dump_buf_end += n;
		i += n;
	}

	return (false);
}

JEMALLOC_ATTR(format(printf, 2, 3))
static bool
prof_printf(bool propagate_err, const char *format, ...)
{
	bool ret;
	va_list ap;
	char buf[PROF_PRINTF_BUFSIZE];

	va_start(ap, format);
	malloc_vsnprintf(buf, sizeof(buf), format, ap);
	va_end(ap);
	ret = prof_write(propagate_err, buf);

	return (ret);
}

static void
prof_ctx_sum(prof_ctx_t *ctx, prof_cnt_t *cnt_all, size_t *leak_nctx)
{
	prof_thr_cnt_t *thr_cnt;
	prof_cnt_t tcnt;

	cassert(config_prof);

	malloc_mutex_lock(ctx->lock);

	memcpy(&ctx->cnt_summed, &ctx->cnt_merged, sizeof(prof_cnt_t));
	ql_foreach(thr_cnt, &ctx->cnts_ql, cnts_link) {
		volatile unsigned *epoch = &thr_cnt->epoch;

		while (true) {
			unsigned epoch0 = *epoch;

			/* Make sure epoch is even. */
			if (epoch0 & 1U)
				continue;

			memcpy(&tcnt, &thr_cnt->cnts, sizeof(prof_cnt_t));

			/* Terminate if epoch didn't change while reading. */
			if (*epoch == epoch0)
				break;
		}

		ctx->cnt_summed.curobjs += tcnt.curobjs;
		ctx->cnt_summed.curbytes += tcnt.curbytes;
		if (opt_prof_accum) {
			ctx->cnt_summed.accumobjs += tcnt.accumobjs;
			ctx->cnt_summed.accumbytes += tcnt.accumbytes;
		}
	}

	if (ctx->cnt_summed.curobjs != 0)
		(*leak_nctx)++;

	/* Add to cnt_all. */
	cnt_all->curobjs += ctx->cnt_summed.curobjs;
	cnt_all->curbytes += ctx->cnt_summed.curbytes;
	if (opt_prof_accum) {
		cnt_all->accumobjs += ctx->cnt_summed.accumobjs;
		cnt_all->accumbytes += ctx->cnt_summed.accumbytes;
	}

	malloc_mutex_unlock(ctx->lock);
}

static void
prof_ctx_destroy(prof_ctx_t *ctx)
{
	prof_tdata_t *prof_tdata;

	cassert(config_prof);

	/*
	 * Check that ctx is still unused by any thread cache before destroying
	 * it.  prof_lookup() increments ctx->nlimbo in order to avoid a race
	 * condition with this function, as does prof_ctx_merge() in order to
	 * avoid a race between the main body of prof_ctx_merge() and entry
	 * into this function.
	 */
	prof_tdata = prof_tdata_get(false);
	assert((uintptr_t)prof_tdata > (uintptr_t)PROF_TDATA_STATE_MAX);
	prof_enter(prof_tdata);
	malloc_mutex_lock(ctx->lock);
	if (ql_first(&ctx->cnts_ql) == NULL && ctx->cnt_merged.curobjs == 0 &&
	    ctx->nlimbo == 1) {
		assert(ctx->cnt_merged.curbytes == 0);
		assert(ctx->cnt_merged.accumobjs == 0);
		assert(ctx->cnt_merged.accumbytes == 0);
		/* Remove ctx from bt2ctx. */
		if (ckh_remove(&bt2ctx, ctx->bt, NULL, NULL))
			assert(false);
		prof_leave(prof_tdata);
		/* Destroy ctx. */
		malloc_mutex_unlock(ctx->lock);
		bt_destroy(ctx->bt);
		idalloc(ctx);
	} else {
		/*
		 * Compensate for increment in prof_ctx_merge() or
		 * prof_lookup().
		 */
		ctx->nlimbo--;
		malloc_mutex_unlock(ctx->lock);
		prof_leave(prof_tdata);
	}
}

static void
prof_ctx_merge(prof_ctx_t *ctx, prof_thr_cnt_t *cnt)
{
	bool destroy;

	cassert(config_prof);

	/* Merge cnt stats and detach from ctx. */
	malloc_mutex_lock(ctx->lock);
	ctx->cnt_merged.curobjs += cnt->cnts.curobjs;
	ctx->cnt_merged.curbytes += cnt->cnts.curbytes;
	ctx->cnt_merged.accumobjs += cnt->cnts.accumobjs;
	ctx->cnt_merged.accumbytes += cnt->cnts.accumbytes;
	ql_remove(&ctx->cnts_ql, cnt, cnts_link);
	if (opt_prof_accum == false && ql_first(&ctx->cnts_ql) == NULL &&
	    ctx->cnt_merged.curobjs == 0 && ctx->nlimbo == 0) {
		/*
		 * Increment ctx->nlimbo in order to keep another thread from
		 * winning the race to destroy ctx while this one has ctx->lock
		 * dropped.  Without this, it would be possible for another
		 * thread to:
		 *
		 * 1) Sample an allocation associated with ctx.
		 * 2) Deallocate the sampled object.
		 * 3) Successfully prof_ctx_destroy(ctx).
		 *
		 * The result would be that ctx no longer exists by the time
		 * this thread accesses it in prof_ctx_destroy().
		 */
		ctx->nlimbo++;
		destroy = true;
	} else
		destroy = false;
	malloc_mutex_unlock(ctx->lock);
	if (destroy)
		prof_ctx_destroy(ctx);
}

static bool
prof_dump_ctx(bool propagate_err, prof_ctx_t *ctx, prof_bt_t *bt)
{
	unsigned i;

	cassert(config_prof);

	/*
	 * Current statistics can sum to 0 as a result of unmerged per thread
	 * statistics.  Additionally, interval- and growth-triggered dumps can
	 * occur between the time a ctx is created and when its statistics are
	 * filled in.  Avoid dumping any ctx that is an artifact of either
	 * implementation detail.
	 */
	if ((opt_prof_accum == false && ctx->cnt_summed.curobjs == 0) ||
	    (opt_prof_accum && ctx->cnt_summed.accumobjs == 0)) {
		assert(ctx->cnt_summed.curobjs == 0);
		assert(ctx->cnt_summed.curbytes == 0);
		assert(ctx->cnt_summed.accumobjs == 0);
		assert(ctx->cnt_summed.accumbytes == 0);
		return (false);
	}

	if (prof_printf(propagate_err, "%"PRId64": %"PRId64
	    " [%"PRIu64": %"PRIu64"] @",
	    ctx->cnt_summed.curobjs, ctx->cnt_summed.curbytes,
	    ctx->cnt_summed.accumobjs, ctx->cnt_summed.accumbytes))
		return (true);

	for (i = 0; i < bt->len; i++) {
		if (prof_printf(propagate_err, " %#"PRIxPTR,
		    (uintptr_t)bt->vec[i]))
			return (true);
	}

	if (prof_write(propagate_err, "\n"))
		return (true);

	return (false);
}

static bool
prof_dump_maps(bool propagate_err)
{
	int mfd;
	char filename[PATH_MAX + 1];

	cassert(config_prof);

	malloc_snprintf(filename, sizeof(filename), "/proc/%d/maps",
	    (int)getpid());
	mfd = open(filename, O_RDONLY);
	if (mfd != -1) {
		ssize_t nread;

		if (prof_write(propagate_err, "\nMAPPED_LIBRARIES:\n") &&
		    propagate_err)
			return (true);
		nread = 0;
		do {
			prof_dump_buf_end += nread;
			if (prof_dump_buf_end == PROF_DUMP_BUFSIZE) {
				/* Make space in prof_dump_buf before read(). */
				if (prof_flush(propagate_err) && propagate_err)
					return (true);
			}
			nread = read(mfd, &prof_dump_buf[prof_dump_buf_end],
			    PROF_DUMP_BUFSIZE - prof_dump_buf_end);
		} while (nread > 0);
		close(mfd);
	} else
		return (true);

	return (false);
}

static bool
prof_dump(bool propagate_err, const char *filename, bool leakcheck)
{
	prof_tdata_t *prof_tdata;
	prof_cnt_t cnt_all;
	size_t tabind;
	union {
		prof_bt_t	*p;
		void		*v;
	} bt;
	union {
		prof_ctx_t	*p;
		void		*v;
	} ctx;
	size_t leak_nctx;

	cassert(config_prof);

	prof_tdata = prof_tdata_get(false);
	if ((uintptr_t)prof_tdata <= (uintptr_t)PROF_TDATA_STATE_MAX)
		return (true);
	prof_enter(prof_tdata);
	prof_dump_fd = creat(filename, 0644);
	if (prof_dump_fd == -1) {
		if (propagate_err == false) {
			malloc_printf(
			    "<jemalloc>: creat(\"%s\"), 0644) failed\n",
			    filename);
			if (opt_abort)
				abort();
		}
		goto label_error;
	}

	/* Merge per thread profile stats, and sum them in cnt_all. */
	memset(&cnt_all, 0, sizeof(prof_cnt_t));
	leak_nctx = 0;
	for (tabind = 0; ckh_iter(&bt2ctx, &tabind, NULL, &ctx.v) == false;)
		prof_ctx_sum(ctx.p, &cnt_all, &leak_nctx);

	/* Dump profile header. */
	if (opt_lg_prof_sample == 0) {
		if (prof_printf(propagate_err,
		    "heap profile: %"PRId64": %"PRId64
		    " [%"PRIu64": %"PRIu64"] @ heapprofile\n",
		    cnt_all.curobjs, cnt_all.curbytes,
		    cnt_all.accumobjs, cnt_all.accumbytes))
			goto label_error;
	} else {
		if (prof_printf(propagate_err,
		    "heap profile: %"PRId64": %"PRId64
		    " [%"PRIu64": %"PRIu64"] @ heap_v2/%"PRIu64"\n",
		    cnt_all.curobjs, cnt_all.curbytes,
		    cnt_all.accumobjs, cnt_all.accumbytes,
		    ((uint64_t)1U << opt_lg_prof_sample)))
			goto label_error;
	}

	/* Dump  per ctx profile stats. */
	for (tabind = 0; ckh_iter(&bt2ctx, &tabind, &bt.v, &ctx.v)
	    == false;) {
		if (prof_dump_ctx(propagate_err, ctx.p, bt.p))
			goto label_error;
	}

	/* Dump /proc/<pid>/maps if possible. */
	if (prof_dump_maps(propagate_err))
		goto label_error;

	if (prof_flush(propagate_err))
		goto label_error;
	close(prof_dump_fd);
	prof_leave(prof_tdata);

	if (leakcheck && cnt_all.curbytes != 0) {
		malloc_printf("<jemalloc>: Leak summary: %"PRId64" byte%s, %"
		    PRId64" object%s, %zu context%s\n",
		    cnt_all.curbytes, (cnt_all.curbytes != 1) ? "s" : "",
		    cnt_all.curobjs, (cnt_all.curobjs != 1) ? "s" : "",
		    leak_nctx, (leak_nctx != 1) ? "s" : "");
		malloc_printf(
		    "<jemalloc>: Run pprof on \"%s\" for leak detail\n",
		    filename);
	}

	return (false);
label_error:
	prof_leave(prof_tdata);
	return (true);
}

#define	DUMP_FILENAME_BUFSIZE	(PATH_MAX + 1)
static void
prof_dump_filename(char *filename, char v, int64_t vseq)
{

	cassert(config_prof);

	if (vseq != UINT64_C(0xffffffffffffffff)) {
	        /* "<prefix>.<pid>.<seq>.v<vseq>.heap" */
		malloc_snprintf(filename, DUMP_FILENAME_BUFSIZE,
		    "%s.%d.%"PRIu64".%c%"PRId64".heap",
		    opt_prof_prefix, (int)getpid(), prof_dump_seq, v, vseq);
	} else {
	        /* "<prefix>.<pid>.<seq>.<v>.heap" */
		malloc_snprintf(filename, DUMP_FILENAME_BUFSIZE,
		    "%s.%d.%"PRIu64".%c.heap",
		    opt_prof_prefix, (int)getpid(), prof_dump_seq, v);
	}
	prof_dump_seq++;
}

static void
prof_fdump(void)
{
	char filename[DUMP_FILENAME_BUFSIZE];

	cassert(config_prof);

	if (prof_booted == false)
		return;

	if (opt_prof_final && opt_prof_prefix[0] != '\0') {
		malloc_mutex_lock(&prof_dump_seq_mtx);
		prof_dump_filename(filename, 'f', UINT64_C(0xffffffffffffffff));
		malloc_mutex_unlock(&prof_dump_seq_mtx);
		prof_dump(false, filename, opt_prof_leak);
	}
}

void
prof_idump(void)
{
	prof_tdata_t *prof_tdata;
	char filename[PATH_MAX + 1];

	cassert(config_prof);

	if (prof_booted == false)
		return;
	prof_tdata = prof_tdata_get(false);
	if ((uintptr_t)prof_tdata <= (uintptr_t)PROF_TDATA_STATE_MAX)
		return;
	if (prof_tdata->enq) {
		prof_tdata->enq_idump = true;
		return;
	}

	if (opt_prof_prefix[0] != '\0') {
		malloc_mutex_lock(&prof_dump_seq_mtx);
		prof_dump_filename(filename, 'i', prof_dump_iseq);
		prof_dump_iseq++;
		malloc_mutex_unlock(&prof_dump_seq_mtx);
		prof_dump(false, filename, false);
	}
}

bool
prof_mdump(const char *filename)
{
	char filename_buf[DUMP_FILENAME_BUFSIZE];

	cassert(config_prof);

	if (opt_prof == false || prof_booted == false)
		return (true);

	if (filename == NULL) {
		/* No filename specified, so automatically generate one. */
		if (opt_prof_prefix[0] == '\0')
			return (true);
		malloc_mutex_lock(&prof_dump_seq_mtx);
		prof_dump_filename(filename_buf, 'm', prof_dump_mseq);
		prof_dump_mseq++;
		malloc_mutex_unlock(&prof_dump_seq_mtx);
		filename = filename_buf;
	}
	return (prof_dump(true, filename, false));
}

void
prof_gdump(void)
{
	prof_tdata_t *prof_tdata;
	char filename[DUMP_FILENAME_BUFSIZE];

	cassert(config_prof);

	if (prof_booted == false)
		return;
	prof_tdata = prof_tdata_get(false);
	if ((uintptr_t)prof_tdata <= (uintptr_t)PROF_TDATA_STATE_MAX)
		return;
	if (prof_tdata->enq) {
		prof_tdata->enq_gdump = true;
		return;
	}

	if (opt_prof_prefix[0] != '\0') {
		malloc_mutex_lock(&prof_dump_seq_mtx);
		prof_dump_filename(filename, 'u', prof_dump_useq);
		prof_dump_useq++;
		malloc_mutex_unlock(&prof_dump_seq_mtx);
		prof_dump(false, filename, false);
	}
}

static void
prof_bt_hash(const void *key, size_t r_hash[2])
{
	prof_bt_t *bt = (prof_bt_t *)key;

	cassert(config_prof);

	hash(bt->vec, bt->len * sizeof(void *), 0x94122f33U, r_hash);
}

static bool
prof_bt_keycomp(const void *k1, const void *k2)
{
	const prof_bt_t *bt1 = (prof_bt_t *)k1;
	const prof_bt_t *bt2 = (prof_bt_t *)k2;

	cassert(config_prof);

	if (bt1->len != bt2->len)
		return (false);
	return (memcmp(bt1->vec, bt2->vec, bt1->len * sizeof(void *)) == 0);
}

static malloc_mutex_t *
prof_ctx_mutex_choose(void)
{
	unsigned nctxs = atomic_add_u(&cum_ctxs, 1);

	return (&ctx_locks[(nctxs - 1) % PROF_NCTX_LOCKS]);
}

prof_tdata_t *
prof_tdata_init(void)
{
	prof_tdata_t *prof_tdata;

	cassert(config_prof);

	/* Initialize an empty cache for this thread. */
	prof_tdata = (prof_tdata_t *)imalloc(sizeof(prof_tdata_t));
	if (prof_tdata == NULL)
		return (NULL);

	if (ckh_new(&prof_tdata->bt2cnt, PROF_CKH_MINITEMS,
	    prof_bt_hash, prof_bt_keycomp)) {
		idalloc(prof_tdata);
		return (NULL);
	}
	ql_new(&prof_tdata->lru_ql);

	prof_tdata->vec = imalloc(sizeof(void *) * PROF_BT_MAX);
	if (prof_tdata->vec == NULL) {
		ckh_delete(&prof_tdata->bt2cnt);
		idalloc(prof_tdata);
		return (NULL);
	}

	prof_tdata->prng_state = 0;
	prof_tdata->threshold = 0;
	prof_tdata->accum = 0;

	prof_tdata->enq = false;
	prof_tdata->enq_idump = false;
	prof_tdata->enq_gdump = false;

	prof_tdata_tsd_set(&prof_tdata);

	return (prof_tdata);
}

void
prof_tdata_cleanup(void *arg)
{
	prof_thr_cnt_t *cnt;
	prof_tdata_t *prof_tdata = *(prof_tdata_t **)arg;

	cassert(config_prof);

	if (prof_tdata == PROF_TDATA_STATE_REINCARNATED) {
		/*
		 * Another destructor deallocated memory after this destructor
		 * was called.  Reset prof_tdata to PROF_TDATA_STATE_PURGATORY
		 * in order to receive another callback.
		 */
		prof_tdata = PROF_TDATA_STATE_PURGATORY;
		prof_tdata_tsd_set(&prof_tdata);
	} else if (prof_tdata == PROF_TDATA_STATE_PURGATORY) {
		/*
		 * The previous time this destructor was called, we set the key
		 * to PROF_TDATA_STATE_PURGATORY so that other destructors
		 * wouldn't cause re-creation of the prof_tdata.  This time, do
		 * nothing, so that the destructor will not be called again.
		 */
	} else if (prof_tdata != NULL) {
		/*
		 * Delete the hash table.  All of its contents can still be
		 * iterated over via the LRU.
		 */
		ckh_delete(&prof_tdata->bt2cnt);
		/*
		 * Iteratively merge cnt's into the global stats and delete
		 * them.
		 */
		while ((cnt = ql_last(&prof_tdata->lru_ql, lru_link)) != NULL) {
			ql_remove(&prof_tdata->lru_ql, cnt, lru_link);
			prof_ctx_merge(cnt->ctx, cnt);
			idalloc(cnt);
		}
		idalloc(prof_tdata->vec);
		idalloc(prof_tdata);
		prof_tdata = PROF_TDATA_STATE_PURGATORY;
		prof_tdata_tsd_set(&prof_tdata);
	}
}

void
prof_boot0(void)
{

	cassert(config_prof);

	memcpy(opt_prof_prefix, PROF_PREFIX_DEFAULT,
	    sizeof(PROF_PREFIX_DEFAULT));
}

void
prof_boot1(void)
{

	cassert(config_prof);

	/*
	 * opt_prof and prof_promote must be in their final state before any
	 * arenas are initialized, so this function must be executed early.
	 */

	if (opt_prof_leak && opt_prof == false) {
		/*
		 * Enable opt_prof, but in such a way that profiles are never
		 * automatically dumped.
		 */
		opt_prof = true;
		opt_prof_gdump = false;
	} else if (opt_prof) {
		if (opt_lg_prof_interval >= 0) {
			prof_interval = (((uint64_t)1U) <<
			    opt_lg_prof_interval);
		}
	}

	prof_promote = (opt_prof && opt_lg_prof_sample > LG_PAGE);
}

bool
prof_boot2(void)
{

	cassert(config_prof);

	if (opt_prof) {
		unsigned i;

		if (ckh_new(&bt2ctx, PROF_CKH_MINITEMS, prof_bt_hash,
		    prof_bt_keycomp))
			return (true);
		if (malloc_mutex_init(&bt2ctx_mtx))
			return (true);
		if (prof_tdata_tsd_boot()) {
			malloc_write(
			    "<jemalloc>: Error in pthread_key_create()\n");
			abort();
		}

		if (malloc_mutex_init(&prof_dump_seq_mtx))
			return (true);

		if (atexit(prof_fdump) != 0) {
			malloc_write("<jemalloc>: Error in atexit()\n");
			if (opt_abort)
				abort();
		}

		ctx_locks = (malloc_mutex_t *)base_alloc(PROF_NCTX_LOCKS *
		    sizeof(malloc_mutex_t));
		if (ctx_locks == NULL)
			return (true);
		for (i = 0; i < PROF_NCTX_LOCKS; i++) {
			if (malloc_mutex_init(&ctx_locks[i]))
				return (true);
		}
	}

#ifdef JEMALLOC_PROF_LIBGCC
	/*
	 * Cause the backtracing machinery to allocate its internal state
	 * before enabling profiling.
	 */
	_Unwind_Backtrace(prof_unwind_init_callback, NULL);
#endif

	prof_booted = true;

	return (false);
}

void
prof_prefork(void)
{

	if (opt_prof) {
		unsigned i;

		malloc_mutex_lock(&bt2ctx_mtx);
		malloc_mutex_lock(&prof_dump_seq_mtx);
		for (i = 0; i < PROF_NCTX_LOCKS; i++)
			malloc_mutex_lock(&ctx_locks[i]);
	}
}

void
prof_postfork_parent(void)
{

	if (opt_prof) {
		unsigned i;

		for (i = 0; i < PROF_NCTX_LOCKS; i++)
			malloc_mutex_postfork_parent(&ctx_locks[i]);
		malloc_mutex_postfork_parent(&prof_dump_seq_mtx);
		malloc_mutex_postfork_parent(&bt2ctx_mtx);
	}
}

void
prof_postfork_child(void)
{

	if (opt_prof) {
		unsigned i;

		for (i = 0; i < PROF_NCTX_LOCKS; i++)
			malloc_mutex_postfork_child(&ctx_locks[i]);
		malloc_mutex_postfork_child(&prof_dump_seq_mtx);
		malloc_mutex_postfork_child(&bt2ctx_mtx);
	}
}

/******************************************************************************/
