#define	JEMALLOC_CTL_C_
#include "jemalloc/internal/jemalloc_internal.h"

/******************************************************************************/
/* Data. */

/*
 * ctl_mtx protects the following:
 * - ctl_stats.*
 * - opt_prof_active
 */
static malloc_mutex_t	ctl_mtx;
static bool		ctl_initialized;
static uint64_t		ctl_epoch;
static ctl_stats_t	ctl_stats;

/******************************************************************************/
/* Helpers for named and indexed nodes. */

static inline const ctl_named_node_t *
ctl_named_node(const ctl_node_t *node)
{

	return ((node->named) ? (const ctl_named_node_t *)node : NULL);
}

static inline const ctl_named_node_t *
ctl_named_children(const ctl_named_node_t *node, int index)
{
	const ctl_named_node_t *children = ctl_named_node(node->children);

	return (children ? &children[index] : NULL);
}

static inline const ctl_indexed_node_t *
ctl_indexed_node(const ctl_node_t *node)
{

	return ((node->named == false) ? (const ctl_indexed_node_t *)node :
	    NULL);
}

/******************************************************************************/
/* Function prototypes for non-inline static functions. */

#define	CTL_PROTO(n)							\
static int	n##_ctl(const size_t *mib, size_t miblen, void *oldp,	\
    size_t *oldlenp, void *newp, size_t newlen);

#define	INDEX_PROTO(n)							\
static const ctl_named_node_t	*n##_index(const size_t *mib,		\
    size_t miblen, size_t i);

static bool	ctl_arena_init(ctl_arena_stats_t *astats);
static void	ctl_arena_clear(ctl_arena_stats_t *astats);
static void	ctl_arena_stats_amerge(ctl_arena_stats_t *cstats,
    arena_t *arena);
static void	ctl_arena_stats_smerge(ctl_arena_stats_t *sstats,
    ctl_arena_stats_t *astats);
static void	ctl_arena_refresh(arena_t *arena, unsigned i);
static bool	ctl_grow(void);
static void	ctl_refresh(void);
static bool	ctl_init(void);
static int	ctl_lookup(const char *name, ctl_node_t const **nodesp,
    size_t *mibp, size_t *depthp);

CTL_PROTO(version)
CTL_PROTO(epoch)
CTL_PROTO(thread_tcache_enabled)
CTL_PROTO(thread_tcache_flush)
CTL_PROTO(thread_arena)
CTL_PROTO(thread_allocated)
CTL_PROTO(thread_allocatedp)
CTL_PROTO(thread_deallocated)
CTL_PROTO(thread_deallocatedp)
CTL_PROTO(config_debug)
CTL_PROTO(config_dss)
CTL_PROTO(config_fill)
CTL_PROTO(config_lazy_lock)
CTL_PROTO(config_mremap)
CTL_PROTO(config_munmap)
CTL_PROTO(config_prof)
CTL_PROTO(config_prof_libgcc)
CTL_PROTO(config_prof_libunwind)
CTL_PROTO(config_stats)
CTL_PROTO(config_tcache)
CTL_PROTO(config_tls)
CTL_PROTO(config_utrace)
CTL_PROTO(config_valgrind)
CTL_PROTO(config_xmalloc)
CTL_PROTO(opt_abort)
CTL_PROTO(opt_dss)
CTL_PROTO(opt_lg_chunk)
CTL_PROTO(opt_narenas)
CTL_PROTO(opt_lg_dirty_mult)
CTL_PROTO(opt_stats_print)
CTL_PROTO(opt_junk)
CTL_PROTO(opt_zero)
CTL_PROTO(opt_quarantine)
CTL_PROTO(opt_redzone)
CTL_PROTO(opt_utrace)
CTL_PROTO(opt_valgrind)
CTL_PROTO(opt_xmalloc)
CTL_PROTO(opt_tcache)
CTL_PROTO(opt_lg_tcache_max)
CTL_PROTO(opt_prof)
CTL_PROTO(opt_prof_prefix)
CTL_PROTO(opt_prof_active)
CTL_PROTO(opt_lg_prof_sample)
CTL_PROTO(opt_lg_prof_interval)
CTL_PROTO(opt_prof_gdump)
CTL_PROTO(opt_prof_final)
CTL_PROTO(opt_prof_leak)
CTL_PROTO(opt_prof_accum)
CTL_PROTO(arena_i_purge)
static void	arena_purge(unsigned arena_ind);
CTL_PROTO(arena_i_dss)
INDEX_PROTO(arena_i)
CTL_PROTO(arenas_bin_i_size)
CTL_PROTO(arenas_bin_i_nregs)
CTL_PROTO(arenas_bin_i_run_size)
INDEX_PROTO(arenas_bin_i)
CTL_PROTO(arenas_lrun_i_size)
INDEX_PROTO(arenas_lrun_i)
CTL_PROTO(arenas_narenas)
CTL_PROTO(arenas_initialized)
CTL_PROTO(arenas_quantum)
CTL_PROTO(arenas_page)
CTL_PROTO(arenas_tcache_max)
CTL_PROTO(arenas_nbins)
CTL_PROTO(arenas_nhbins)
CTL_PROTO(arenas_nlruns)
CTL_PROTO(arenas_purge)
CTL_PROTO(arenas_extend)
CTL_PROTO(prof_active)
CTL_PROTO(prof_dump)
CTL_PROTO(prof_interval)
CTL_PROTO(stats_chunks_current)
CTL_PROTO(stats_chunks_total)
CTL_PROTO(stats_chunks_high)
CTL_PROTO(stats_huge_allocated)
CTL_PROTO(stats_huge_nmalloc)
CTL_PROTO(stats_huge_ndalloc)
CTL_PROTO(stats_arenas_i_small_allocated)
CTL_PROTO(stats_arenas_i_small_nmalloc)
CTL_PROTO(stats_arenas_i_small_ndalloc)
CTL_PROTO(stats_arenas_i_small_nrequests)
CTL_PROTO(stats_arenas_i_large_allocated)
CTL_PROTO(stats_arenas_i_large_nmalloc)
CTL_PROTO(stats_arenas_i_large_ndalloc)
CTL_PROTO(stats_arenas_i_large_nrequests)
CTL_PROTO(stats_arenas_i_bins_j_allocated)
CTL_PROTO(stats_arenas_i_bins_j_nmalloc)
CTL_PROTO(stats_arenas_i_bins_j_ndalloc)
CTL_PROTO(stats_arenas_i_bins_j_nrequests)
CTL_PROTO(stats_arenas_i_bins_j_nfills)
CTL_PROTO(stats_arenas_i_bins_j_nflushes)
CTL_PROTO(stats_arenas_i_bins_j_nruns)
CTL_PROTO(stats_arenas_i_bins_j_nreruns)
CTL_PROTO(stats_arenas_i_bins_j_curruns)
INDEX_PROTO(stats_arenas_i_bins_j)
CTL_PROTO(stats_arenas_i_lruns_j_nmalloc)
CTL_PROTO(stats_arenas_i_lruns_j_ndalloc)
CTL_PROTO(stats_arenas_i_lruns_j_nrequests)
CTL_PROTO(stats_arenas_i_lruns_j_curruns)
INDEX_PROTO(stats_arenas_i_lruns_j)
CTL_PROTO(stats_arenas_i_nthreads)
CTL_PROTO(stats_arenas_i_dss)
CTL_PROTO(stats_arenas_i_pactive)
CTL_PROTO(stats_arenas_i_pdirty)
CTL_PROTO(stats_arenas_i_mapped)
CTL_PROTO(stats_arenas_i_npurge)
CTL_PROTO(stats_arenas_i_nmadvise)
CTL_PROTO(stats_arenas_i_purged)
INDEX_PROTO(stats_arenas_i)
CTL_PROTO(stats_cactive)
CTL_PROTO(stats_allocated)
CTL_PROTO(stats_active)
CTL_PROTO(stats_mapped)

/******************************************************************************/
/* mallctl tree. */

/* Maximum tree depth. */
#define	CTL_MAX_DEPTH	6

#define	NAME(n)	{true},	n
#define	CHILD(t, c)							\
	sizeof(c##_node) / sizeof(ctl_##t##_node_t),			\
	(ctl_node_t *)c##_node,						\
	NULL
#define	CTL(c)	0, NULL, c##_ctl

/*
 * Only handles internal indexed nodes, since there are currently no external
 * ones.
 */
#define	INDEX(i)	{false},	i##_index

static const ctl_named_node_t	tcache_node[] = {
	{NAME("enabled"),	CTL(thread_tcache_enabled)},
	{NAME("flush"),		CTL(thread_tcache_flush)}
};

static const ctl_named_node_t	thread_node[] = {
	{NAME("arena"),		CTL(thread_arena)},
	{NAME("allocated"),	CTL(thread_allocated)},
	{NAME("allocatedp"),	CTL(thread_allocatedp)},
	{NAME("deallocated"),	CTL(thread_deallocated)},
	{NAME("deallocatedp"),	CTL(thread_deallocatedp)},
	{NAME("tcache"),	CHILD(named, tcache)}
};

static const ctl_named_node_t	config_node[] = {
	{NAME("debug"),			CTL(config_debug)},
	{NAME("dss"),			CTL(config_dss)},
	{NAME("fill"),			CTL(config_fill)},
	{NAME("lazy_lock"),		CTL(config_lazy_lock)},
	{NAME("mremap"),		CTL(config_mremap)},
	{NAME("munmap"),		CTL(config_munmap)},
	{NAME("prof"),			CTL(config_prof)},
	{NAME("prof_libgcc"),		CTL(config_prof_libgcc)},
	{NAME("prof_libunwind"),	CTL(config_prof_libunwind)},
	{NAME("stats"),			CTL(config_stats)},
	{NAME("tcache"),		CTL(config_tcache)},
	{NAME("tls"),			CTL(config_tls)},
	{NAME("utrace"),		CTL(config_utrace)},
	{NAME("valgrind"),		CTL(config_valgrind)},
	{NAME("xmalloc"),		CTL(config_xmalloc)}
};

static const ctl_named_node_t opt_node[] = {
	{NAME("abort"),			CTL(opt_abort)},
	{NAME("dss"),			CTL(opt_dss)},
	{NAME("lg_chunk"),		CTL(opt_lg_chunk)},
	{NAME("narenas"),		CTL(opt_narenas)},
	{NAME("lg_dirty_mult"),		CTL(opt_lg_dirty_mult)},
	{NAME("stats_print"),		CTL(opt_stats_print)},
	{NAME("junk"),			CTL(opt_junk)},
	{NAME("zero"),			CTL(opt_zero)},
	{NAME("quarantine"),		CTL(opt_quarantine)},
	{NAME("redzone"),		CTL(opt_redzone)},
	{NAME("utrace"),		CTL(opt_utrace)},
	{NAME("valgrind"),		CTL(opt_valgrind)},
	{NAME("xmalloc"),		CTL(opt_xmalloc)},
	{NAME("tcache"),		CTL(opt_tcache)},
	{NAME("lg_tcache_max"),		CTL(opt_lg_tcache_max)},
	{NAME("prof"),			CTL(opt_prof)},
	{NAME("prof_prefix"),		CTL(opt_prof_prefix)},
	{NAME("prof_active"),		CTL(opt_prof_active)},
	{NAME("lg_prof_sample"),	CTL(opt_lg_prof_sample)},
	{NAME("lg_prof_interval"),	CTL(opt_lg_prof_interval)},
	{NAME("prof_gdump"),		CTL(opt_prof_gdump)},
	{NAME("prof_final"),		CTL(opt_prof_final)},
	{NAME("prof_leak"),		CTL(opt_prof_leak)},
	{NAME("prof_accum"),		CTL(opt_prof_accum)}
};

static const ctl_named_node_t arena_i_node[] = {
	{NAME("purge"),			CTL(arena_i_purge)},
	{NAME("dss"),			CTL(arena_i_dss)}
};
static const ctl_named_node_t super_arena_i_node[] = {
	{NAME(""),			CHILD(named, arena_i)}
};

static const ctl_indexed_node_t arena_node[] = {
	{INDEX(arena_i)}
};

static const ctl_named_node_t arenas_bin_i_node[] = {
	{NAME("size"),			CTL(arenas_bin_i_size)},
	{NAME("nregs"),			CTL(arenas_bin_i_nregs)},
	{NAME("run_size"),		CTL(arenas_bin_i_run_size)}
};
static const ctl_named_node_t super_arenas_bin_i_node[] = {
	{NAME(""),			CHILD(named, arenas_bin_i)}
};

static const ctl_indexed_node_t arenas_bin_node[] = {
	{INDEX(arenas_bin_i)}
};

static const ctl_named_node_t arenas_lrun_i_node[] = {
	{NAME("size"),			CTL(arenas_lrun_i_size)}
};
static const ctl_named_node_t super_arenas_lrun_i_node[] = {
	{NAME(""),			CHILD(named, arenas_lrun_i)}
};

static const ctl_indexed_node_t arenas_lrun_node[] = {
	{INDEX(arenas_lrun_i)}
};

static const ctl_named_node_t arenas_node[] = {
	{NAME("narenas"),		CTL(arenas_narenas)},
	{NAME("initialized"),		CTL(arenas_initialized)},
	{NAME("quantum"),		CTL(arenas_quantum)},
	{NAME("page"),			CTL(arenas_page)},
	{NAME("tcache_max"),		CTL(arenas_tcache_max)},
	{NAME("nbins"),			CTL(arenas_nbins)},
	{NAME("nhbins"),		CTL(arenas_nhbins)},
	{NAME("bin"),			CHILD(indexed, arenas_bin)},
	{NAME("nlruns"),		CTL(arenas_nlruns)},
	{NAME("lrun"),			CHILD(indexed, arenas_lrun)},
	{NAME("purge"),			CTL(arenas_purge)},
	{NAME("extend"),		CTL(arenas_extend)}
};

static const ctl_named_node_t	prof_node[] = {
	{NAME("active"),	CTL(prof_active)},
	{NAME("dump"),		CTL(prof_dump)},
	{NAME("interval"),	CTL(prof_interval)}
};

static const ctl_named_node_t stats_chunks_node[] = {
	{NAME("current"),		CTL(stats_chunks_current)},
	{NAME("total"),			CTL(stats_chunks_total)},
	{NAME("high"),			CTL(stats_chunks_high)}
};

static const ctl_named_node_t stats_huge_node[] = {
	{NAME("allocated"),		CTL(stats_huge_allocated)},
	{NAME("nmalloc"),		CTL(stats_huge_nmalloc)},
	{NAME("ndalloc"),		CTL(stats_huge_ndalloc)}
};

static const ctl_named_node_t stats_arenas_i_small_node[] = {
	{NAME("allocated"),		CTL(stats_arenas_i_small_allocated)},
	{NAME("nmalloc"),		CTL(stats_arenas_i_small_nmalloc)},
	{NAME("ndalloc"),		CTL(stats_arenas_i_small_ndalloc)},
	{NAME("nrequests"),		CTL(stats_arenas_i_small_nrequests)}
};

static const ctl_named_node_t stats_arenas_i_large_node[] = {
	{NAME("allocated"),		CTL(stats_arenas_i_large_allocated)},
	{NAME("nmalloc"),		CTL(stats_arenas_i_large_nmalloc)},
	{NAME("ndalloc"),		CTL(stats_arenas_i_large_ndalloc)},
	{NAME("nrequests"),		CTL(stats_arenas_i_large_nrequests)}
};

static const ctl_named_node_t stats_arenas_i_bins_j_node[] = {
	{NAME("allocated"),		CTL(stats_arenas_i_bins_j_allocated)},
	{NAME("nmalloc"),		CTL(stats_arenas_i_bins_j_nmalloc)},
	{NAME("ndalloc"),		CTL(stats_arenas_i_bins_j_ndalloc)},
	{NAME("nrequests"),		CTL(stats_arenas_i_bins_j_nrequests)},
	{NAME("nfills"),		CTL(stats_arenas_i_bins_j_nfills)},
	{NAME("nflushes"),		CTL(stats_arenas_i_bins_j_nflushes)},
	{NAME("nruns"),			CTL(stats_arenas_i_bins_j_nruns)},
	{NAME("nreruns"),		CTL(stats_arenas_i_bins_j_nreruns)},
	{NAME("curruns"),		CTL(stats_arenas_i_bins_j_curruns)}
};
static const ctl_named_node_t super_stats_arenas_i_bins_j_node[] = {
	{NAME(""),			CHILD(named, stats_arenas_i_bins_j)}
};

static const ctl_indexed_node_t stats_arenas_i_bins_node[] = {
	{INDEX(stats_arenas_i_bins_j)}
};

static const ctl_named_node_t stats_arenas_i_lruns_j_node[] = {
	{NAME("nmalloc"),		CTL(stats_arenas_i_lruns_j_nmalloc)},
	{NAME("ndalloc"),		CTL(stats_arenas_i_lruns_j_ndalloc)},
	{NAME("nrequests"),		CTL(stats_arenas_i_lruns_j_nrequests)},
	{NAME("curruns"),		CTL(stats_arenas_i_lruns_j_curruns)}
};
static const ctl_named_node_t super_stats_arenas_i_lruns_j_node[] = {
	{NAME(""),			CHILD(named, stats_arenas_i_lruns_j)}
};

static const ctl_indexed_node_t stats_arenas_i_lruns_node[] = {
	{INDEX(stats_arenas_i_lruns_j)}
};

static const ctl_named_node_t stats_arenas_i_node[] = {
	{NAME("nthreads"),		CTL(stats_arenas_i_nthreads)},
	{NAME("dss"),			CTL(stats_arenas_i_dss)},
	{NAME("pactive"),		CTL(stats_arenas_i_pactive)},
	{NAME("pdirty"),		CTL(stats_arenas_i_pdirty)},
	{NAME("mapped"),		CTL(stats_arenas_i_mapped)},
	{NAME("npurge"),		CTL(stats_arenas_i_npurge)},
	{NAME("nmadvise"),		CTL(stats_arenas_i_nmadvise)},
	{NAME("purged"),		CTL(stats_arenas_i_purged)},
	{NAME("small"),			CHILD(named, stats_arenas_i_small)},
	{NAME("large"),			CHILD(named, stats_arenas_i_large)},
	{NAME("bins"),			CHILD(indexed, stats_arenas_i_bins)},
	{NAME("lruns"),			CHILD(indexed, stats_arenas_i_lruns)}
};
static const ctl_named_node_t super_stats_arenas_i_node[] = {
	{NAME(""),			CHILD(named, stats_arenas_i)}
};

static const ctl_indexed_node_t stats_arenas_node[] = {
	{INDEX(stats_arenas_i)}
};

static const ctl_named_node_t stats_node[] = {
	{NAME("cactive"),		CTL(stats_cactive)},
	{NAME("allocated"),		CTL(stats_allocated)},
	{NAME("active"),		CTL(stats_active)},
	{NAME("mapped"),		CTL(stats_mapped)},
	{NAME("chunks"),		CHILD(named, stats_chunks)},
	{NAME("huge"),			CHILD(named, stats_huge)},
	{NAME("arenas"),		CHILD(indexed, stats_arenas)}
};

static const ctl_named_node_t	root_node[] = {
	{NAME("version"),	CTL(version)},
	{NAME("epoch"),		CTL(epoch)},
	{NAME("thread"),	CHILD(named, thread)},
	{NAME("config"),	CHILD(named, config)},
	{NAME("opt"),		CHILD(named, opt)},
	{NAME("arena"),		CHILD(indexed, arena)},
	{NAME("arenas"),	CHILD(named, arenas)},
	{NAME("prof"),		CHILD(named, prof)},
	{NAME("stats"),		CHILD(named, stats)}
};
static const ctl_named_node_t super_root_node[] = {
	{NAME(""),		CHILD(named, root)}
};

#undef NAME
#undef CHILD
#undef CTL
#undef INDEX

/******************************************************************************/

static bool
ctl_arena_init(ctl_arena_stats_t *astats)
{

	if (astats->lstats == NULL) {
		astats->lstats = (malloc_large_stats_t *)base_alloc(nlclasses *
		    sizeof(malloc_large_stats_t));
		if (astats->lstats == NULL)
			return (true);
	}

	return (false);
}

static void
ctl_arena_clear(ctl_arena_stats_t *astats)
{

	astats->dss = dss_prec_names[dss_prec_limit];
	astats->pactive = 0;
	astats->pdirty = 0;
	if (config_stats) {
		memset(&astats->astats, 0, sizeof(arena_stats_t));
		astats->allocated_small = 0;
		astats->nmalloc_small = 0;
		astats->ndalloc_small = 0;
		astats->nrequests_small = 0;
		memset(astats->bstats, 0, NBINS * sizeof(malloc_bin_stats_t));
		memset(astats->lstats, 0, nlclasses *
		    sizeof(malloc_large_stats_t));
	}
}

static void
ctl_arena_stats_amerge(ctl_arena_stats_t *cstats, arena_t *arena)
{
	unsigned i;

	arena_stats_merge(arena, &cstats->dss, &cstats->pactive,
	    &cstats->pdirty, &cstats->astats, cstats->bstats, cstats->lstats);

	for (i = 0; i < NBINS; i++) {
		cstats->allocated_small += cstats->bstats[i].allocated;
		cstats->nmalloc_small += cstats->bstats[i].nmalloc;
		cstats->ndalloc_small += cstats->bstats[i].ndalloc;
		cstats->nrequests_small += cstats->bstats[i].nrequests;
	}
}

static void
ctl_arena_stats_smerge(ctl_arena_stats_t *sstats, ctl_arena_stats_t *astats)
{
	unsigned i;

	sstats->pactive += astats->pactive;
	sstats->pdirty += astats->pdirty;

	sstats->astats.mapped += astats->astats.mapped;
	sstats->astats.npurge += astats->astats.npurge;
	sstats->astats.nmadvise += astats->astats.nmadvise;
	sstats->astats.purged += astats->astats.purged;

	sstats->allocated_small += astats->allocated_small;
	sstats->nmalloc_small += astats->nmalloc_small;
	sstats->ndalloc_small += astats->ndalloc_small;
	sstats->nrequests_small += astats->nrequests_small;

	sstats->astats.allocated_large += astats->astats.allocated_large;
	sstats->astats.nmalloc_large += astats->astats.nmalloc_large;
	sstats->astats.ndalloc_large += astats->astats.ndalloc_large;
	sstats->astats.nrequests_large += astats->astats.nrequests_large;

	for (i = 0; i < nlclasses; i++) {
		sstats->lstats[i].nmalloc += astats->lstats[i].nmalloc;
		sstats->lstats[i].ndalloc += astats->lstats[i].ndalloc;
		sstats->lstats[i].nrequests += astats->lstats[i].nrequests;
		sstats->lstats[i].curruns += astats->lstats[i].curruns;
	}

	for (i = 0; i < NBINS; i++) {
		sstats->bstats[i].allocated += astats->bstats[i].allocated;
		sstats->bstats[i].nmalloc += astats->bstats[i].nmalloc;
		sstats->bstats[i].ndalloc += astats->bstats[i].ndalloc;
		sstats->bstats[i].nrequests += astats->bstats[i].nrequests;
		if (config_tcache) {
			sstats->bstats[i].nfills += astats->bstats[i].nfills;
			sstats->bstats[i].nflushes +=
			    astats->bstats[i].nflushes;
		}
		sstats->bstats[i].nruns += astats->bstats[i].nruns;
		sstats->bstats[i].reruns += astats->bstats[i].reruns;
		sstats->bstats[i].curruns += astats->bstats[i].curruns;
	}
}

static void
ctl_arena_refresh(arena_t *arena, unsigned i)
{
	ctl_arena_stats_t *astats = &ctl_stats.arenas[i];
	ctl_arena_stats_t *sstats = &ctl_stats.arenas[ctl_stats.narenas];

	ctl_arena_clear(astats);

	sstats->nthreads += astats->nthreads;
	if (config_stats) {
		ctl_arena_stats_amerge(astats, arena);
		/* Merge into sum stats as well. */
		ctl_arena_stats_smerge(sstats, astats);
	} else {
		astats->pactive += arena->nactive;
		astats->pdirty += arena->ndirty;
		/* Merge into sum stats as well. */
		sstats->pactive += arena->nactive;
		sstats->pdirty += arena->ndirty;
	}
}

static bool
ctl_grow(void)
{
	size_t astats_size;
	ctl_arena_stats_t *astats;
	arena_t **tarenas;

	/* Extend arena stats and arenas arrays. */
	astats_size = (ctl_stats.narenas + 2) * sizeof(ctl_arena_stats_t);
	if (ctl_stats.narenas == narenas_auto) {
		/* ctl_stats.arenas and arenas came from base_alloc(). */
		astats = (ctl_arena_stats_t *)imalloc(astats_size);
		if (astats == NULL)
			return (true);
		memcpy(astats, ctl_stats.arenas, (ctl_stats.narenas + 1) *
		    sizeof(ctl_arena_stats_t));

		tarenas = (arena_t **)imalloc((ctl_stats.narenas + 1) *
		    sizeof(arena_t *));
		if (tarenas == NULL) {
			idalloc(astats);
			return (true);
		}
		memcpy(tarenas, arenas, ctl_stats.narenas * sizeof(arena_t *));
	} else {
		astats = (ctl_arena_stats_t *)iralloc(ctl_stats.arenas,
		    astats_size, 0, 0, false, false);
		if (astats == NULL)
			return (true);

		tarenas = (arena_t **)iralloc(arenas, (ctl_stats.narenas + 1) *
		    sizeof(arena_t *), 0, 0, false, false);
		if (tarenas == NULL)
			return (true);
	}
	/* Initialize the new astats and arenas elements. */
	memset(&astats[ctl_stats.narenas + 1], 0, sizeof(ctl_arena_stats_t));
	if (ctl_arena_init(&astats[ctl_stats.narenas + 1]))
		return (true);
	tarenas[ctl_stats.narenas] = NULL;
	/* Swap merged stats to their new location. */
	{
		ctl_arena_stats_t tstats;
		memcpy(&tstats, &astats[ctl_stats.narenas],
		    sizeof(ctl_arena_stats_t));
		memcpy(&astats[ctl_stats.narenas],
		    &astats[ctl_stats.narenas + 1], sizeof(ctl_arena_stats_t));
		memcpy(&astats[ctl_stats.narenas + 1], &tstats,
		    sizeof(ctl_arena_stats_t));
	}
	ctl_stats.arenas = astats;
	ctl_stats.narenas++;
	malloc_mutex_lock(&arenas_lock);
	arenas = tarenas;
	narenas_total++;
	arenas_extend(narenas_total - 1);
	malloc_mutex_unlock(&arenas_lock);

	return (false);
}

static void
ctl_refresh(void)
{
	unsigned i;
	VARIABLE_ARRAY(arena_t *, tarenas, ctl_stats.narenas);

	if (config_stats) {
		malloc_mutex_lock(&chunks_mtx);
		ctl_stats.chunks.current = stats_chunks.curchunks;
		ctl_stats.chunks.total = stats_chunks.nchunks;
		ctl_stats.chunks.high = stats_chunks.highchunks;
		malloc_mutex_unlock(&chunks_mtx);

		malloc_mutex_lock(&huge_mtx);
		ctl_stats.huge.allocated = huge_allocated;
		ctl_stats.huge.nmalloc = huge_nmalloc;
		ctl_stats.huge.ndalloc = huge_ndalloc;
		malloc_mutex_unlock(&huge_mtx);
	}

	/*
	 * Clear sum stats, since they will be merged into by
	 * ctl_arena_refresh().
	 */
	ctl_stats.arenas[ctl_stats.narenas].nthreads = 0;
	ctl_arena_clear(&ctl_stats.arenas[ctl_stats.narenas]);

	malloc_mutex_lock(&arenas_lock);
	memcpy(tarenas, arenas, sizeof(arena_t *) * ctl_stats.narenas);
	for (i = 0; i < ctl_stats.narenas; i++) {
		if (arenas[i] != NULL)
			ctl_stats.arenas[i].nthreads = arenas[i]->nthreads;
		else
			ctl_stats.arenas[i].nthreads = 0;
	}
	malloc_mutex_unlock(&arenas_lock);
	for (i = 0; i < ctl_stats.narenas; i++) {
		bool initialized = (tarenas[i] != NULL);

		ctl_stats.arenas[i].initialized = initialized;
		if (initialized)
			ctl_arena_refresh(tarenas[i], i);
	}

	if (config_stats) {
		ctl_stats.allocated =
		    ctl_stats.arenas[ctl_stats.narenas].allocated_small
		    + ctl_stats.arenas[ctl_stats.narenas].astats.allocated_large
		    + ctl_stats.huge.allocated;
		ctl_stats.active =
		    (ctl_stats.arenas[ctl_stats.narenas].pactive << LG_PAGE)
		    + ctl_stats.huge.allocated;
		ctl_stats.mapped = (ctl_stats.chunks.current << opt_lg_chunk);
	}

	ctl_epoch++;
}

static bool
ctl_init(void)
{
	bool ret;

	malloc_mutex_lock(&ctl_mtx);
	if (ctl_initialized == false) {
		/*
		 * Allocate space for one extra arena stats element, which
		 * contains summed stats across all arenas.
		 */
		assert(narenas_auto == narenas_total_get());
		ctl_stats.narenas = narenas_auto;
		ctl_stats.arenas = (ctl_arena_stats_t *)base_alloc(
		    (ctl_stats.narenas + 1) * sizeof(ctl_arena_stats_t));
		if (ctl_stats.arenas == NULL) {
			ret = true;
			goto label_return;
		}
		memset(ctl_stats.arenas, 0, (ctl_stats.narenas + 1) *
		    sizeof(ctl_arena_stats_t));

		/*
		 * Initialize all stats structures, regardless of whether they
		 * ever get used.  Lazy initialization would allow errors to
		 * cause inconsistent state to be viewable by the application.
		 */
		if (config_stats) {
			unsigned i;
			for (i = 0; i <= ctl_stats.narenas; i++) {
				if (ctl_arena_init(&ctl_stats.arenas[i])) {
					ret = true;
					goto label_return;
				}
			}
		}
		ctl_stats.arenas[ctl_stats.narenas].initialized = true;

		ctl_epoch = 0;
		ctl_refresh();
		ctl_initialized = true;
	}

	ret = false;
label_return:
	malloc_mutex_unlock(&ctl_mtx);
	return (ret);
}

static int
ctl_lookup(const char *name, ctl_node_t const **nodesp, size_t *mibp,
    size_t *depthp)
{
	int ret;
	const char *elm, *tdot, *dot;
	size_t elen, i, j;
	const ctl_named_node_t *node;

	elm = name;
	/* Equivalent to strchrnul(). */
	dot = ((tdot = strchr(elm, '.')) != NULL) ? tdot : strchr(elm, '\0');
	elen = (size_t)((uintptr_t)dot - (uintptr_t)elm);
	if (elen == 0) {
		ret = ENOENT;
		goto label_return;
	}
	node = super_root_node;
	for (i = 0; i < *depthp; i++) {
		assert(node);
		assert(node->nchildren > 0);
		if (ctl_named_node(node->children) != NULL) {
			const ctl_named_node_t *pnode = node;

			/* Children are named. */
			for (j = 0; j < node->nchildren; j++) {
				const ctl_named_node_t *child =
				    ctl_named_children(node, j);
				if (strlen(child->name) == elen &&
				    strncmp(elm, child->name, elen) == 0) {
					node = child;
					if (nodesp != NULL)
						nodesp[i] =
						    (const ctl_node_t *)node;
					mibp[i] = j;
					break;
				}
			}
			if (node == pnode) {
				ret = ENOENT;
				goto label_return;
			}
		} else {
			uintmax_t index;
			const ctl_indexed_node_t *inode;

			/* Children are indexed. */
			index = malloc_strtoumax(elm, NULL, 10);
			if (index == UINTMAX_MAX || index > SIZE_T_MAX) {
				ret = ENOENT;
				goto label_return;
			}

			inode = ctl_indexed_node(node->children);
			node = inode->index(mibp, *depthp, (size_t)index);
			if (node == NULL) {
				ret = ENOENT;
				goto label_return;
			}

			if (nodesp != NULL)
				nodesp[i] = (const ctl_node_t *)node;
			mibp[i] = (size_t)index;
		}

		if (node->ctl != NULL) {
			/* Terminal node. */
			if (*dot != '\0') {
				/*
				 * The name contains more elements than are
				 * in this path through the tree.
				 */
				ret = ENOENT;
				goto label_return;
			}
			/* Complete lookup successful. */
			*depthp = i + 1;
			break;
		}

		/* Update elm. */
		if (*dot == '\0') {
			/* No more elements. */
			ret = ENOENT;
			goto label_return;
		}
		elm = &dot[1];
		dot = ((tdot = strchr(elm, '.')) != NULL) ? tdot :
		    strchr(elm, '\0');
		elen = (size_t)((uintptr_t)dot - (uintptr_t)elm);
	}

	ret = 0;
label_return:
	return (ret);
}

int
ctl_byname(const char *name, void *oldp, size_t *oldlenp, void *newp,
    size_t newlen)
{
	int ret;
	size_t depth;
	ctl_node_t const *nodes[CTL_MAX_DEPTH];
	size_t mib[CTL_MAX_DEPTH];
	const ctl_named_node_t *node;

	if (ctl_initialized == false && ctl_init()) {
		ret = EAGAIN;
		goto label_return;
	}

	depth = CTL_MAX_DEPTH;
	ret = ctl_lookup(name, nodes, mib, &depth);
	if (ret != 0)
		goto label_return;

	node = ctl_named_node(nodes[depth-1]);
	if (node != NULL && node->ctl)
		ret = node->ctl(mib, depth, oldp, oldlenp, newp, newlen);
	else {
		/* The name refers to a partial path through the ctl tree. */
		ret = ENOENT;
	}

label_return:
	return(ret);
}

int
ctl_nametomib(const char *name, size_t *mibp, size_t *miblenp)
{
	int ret;

	if (ctl_initialized == false && ctl_init()) {
		ret = EAGAIN;
		goto label_return;
	}

	ret = ctl_lookup(name, NULL, mibp, miblenp);
label_return:
	return(ret);
}

int
ctl_bymib(const size_t *mib, size_t miblen, void *oldp, size_t *oldlenp,
    void *newp, size_t newlen)
{
	int ret;
	const ctl_named_node_t *node;
	size_t i;

	if (ctl_initialized == false && ctl_init()) {
		ret = EAGAIN;
		goto label_return;
	}

	/* Iterate down the tree. */
	node = super_root_node;
	for (i = 0; i < miblen; i++) {
		assert(node);
		assert(node->nchildren > 0);
		if (ctl_named_node(node->children) != NULL) {
			/* Children are named. */
			if (node->nchildren <= mib[i]) {
				ret = ENOENT;
				goto label_return;
			}
			node = ctl_named_children(node, mib[i]);
		} else {
			const ctl_indexed_node_t *inode;

			/* Indexed element. */
			inode = ctl_indexed_node(node->children);
			node = inode->index(mib, miblen, mib[i]);
			if (node == NULL) {
				ret = ENOENT;
				goto label_return;
			}
		}
	}

	/* Call the ctl function. */
	if (node && node->ctl)
		ret = node->ctl(mib, miblen, oldp, oldlenp, newp, newlen);
	else {
		/* Partial MIB. */
		ret = ENOENT;
	}

label_return:
	return(ret);
}

bool
ctl_boot(void)
{

	if (malloc_mutex_init(&ctl_mtx))
		return (true);

	ctl_initialized = false;

	return (false);
}

void
ctl_prefork(void)
{

	malloc_mutex_lock(&ctl_mtx);
}

void
ctl_postfork_parent(void)
{

	malloc_mutex_postfork_parent(&ctl_mtx);
}

void
ctl_postfork_child(void)
{

	malloc_mutex_postfork_child(&ctl_mtx);
}

/******************************************************************************/
/* *_ctl() functions. */

#define	READONLY()	do {						\
	if (newp != NULL || newlen != 0) {				\
		ret = EPERM;						\
		goto label_return;					\
	}								\
} while (0)

#define	WRITEONLY()	do {						\
	if (oldp != NULL || oldlenp != NULL) {				\
		ret = EPERM;						\
		goto label_return;					\
	}								\
} while (0)

#define	READ(v, t)	do {						\
	if (oldp != NULL && oldlenp != NULL) {				\
		if (*oldlenp != sizeof(t)) {				\
			size_t	copylen = (sizeof(t) <= *oldlenp)	\
			    ? sizeof(t) : *oldlenp;			\
			memcpy(oldp, (void *)&(v), copylen);		\
			ret = EINVAL;					\
			goto label_return;				\
		} else							\
			*(t *)oldp = (v);				\
	}								\
} while (0)

#define	WRITE(v, t)	do {						\
	if (newp != NULL) {						\
		if (newlen != sizeof(t)) {				\
			ret = EINVAL;					\
			goto label_return;				\
		}							\
		(v) = *(t *)newp;					\
	}								\
} while (0)

/*
 * There's a lot of code duplication in the following macros due to limitations
 * in how nested cpp macros are expanded.
 */
#define	CTL_RO_CLGEN(c, l, n, v, t)					\
static int								\
n##_ctl(const size_t *mib, size_t miblen, void *oldp, size_t *oldlenp,	\
    void *newp, size_t newlen)						\
{									\
	int ret;							\
	t oldval;							\
									\
	if ((c) == false)						\
		return (ENOENT);					\
	if (l)								\
		malloc_mutex_lock(&ctl_mtx);				\
	READONLY();							\
	oldval = (v);							\
	READ(oldval, t);						\
									\
	ret = 0;							\
label_return:								\
	if (l)								\
		malloc_mutex_unlock(&ctl_mtx);				\
	return (ret);							\
}

#define	CTL_RO_CGEN(c, n, v, t)						\
static int								\
n##_ctl(const size_t *mib, size_t miblen, void *oldp, size_t *oldlenp,	\
    void *newp, size_t newlen)						\
{									\
	int ret;							\
	t oldval;							\
									\
	if ((c) == false)						\
		return (ENOENT);					\
	malloc_mutex_lock(&ctl_mtx);					\
	READONLY();							\
	oldval = (v);							\
	READ(oldval, t);						\
									\
	ret = 0;							\
label_return:								\
	malloc_mutex_unlock(&ctl_mtx);					\
	return (ret);							\
}

#define	CTL_RO_GEN(n, v, t)						\
static int								\
n##_ctl(const size_t *mib, size_t miblen, void *oldp, size_t *oldlenp,	\
    void *newp, size_t newlen)						\
{									\
	int ret;							\
	t oldval;							\
									\
	malloc_mutex_lock(&ctl_mtx);					\
	READONLY();							\
	oldval = (v);							\
	READ(oldval, t);						\
									\
	ret = 0;							\
label_return:								\
	malloc_mutex_unlock(&ctl_mtx);					\
	return (ret);							\
}

/*
 * ctl_mtx is not acquired, under the assumption that no pertinent data will
 * mutate during the call.
 */
#define	CTL_RO_NL_CGEN(c, n, v, t)					\
static int								\
n##_ctl(const size_t *mib, size_t miblen, void *oldp, size_t *oldlenp,	\
    void *newp, size_t newlen)						\
{									\
	int ret;							\
	t oldval;							\
									\
	if ((c) == false)						\
		return (ENOENT);					\
	READONLY();							\
	oldval = (v);							\
	READ(oldval, t);						\
									\
	ret = 0;							\
label_return:								\
	return (ret);							\
}

#define	CTL_RO_NL_GEN(n, v, t)						\
static int								\
n##_ctl(const size_t *mib, size_t miblen, void *oldp, size_t *oldlenp,	\
    void *newp, size_t newlen)						\
{									\
	int ret;							\
	t oldval;							\
									\
	READONLY();							\
	oldval = (v);							\
	READ(oldval, t);						\
									\
	ret = 0;							\
label_return:								\
	return (ret);							\
}

#define	CTL_RO_BOOL_CONFIG_GEN(n)					\
static int								\
n##_ctl(const size_t *mib, size_t miblen, void *oldp, size_t *oldlenp,	\
    void *newp, size_t newlen)						\
{									\
	int ret;							\
	bool oldval;							\
									\
	READONLY();							\
	oldval = n;							\
	READ(oldval, bool);						\
									\
	ret = 0;							\
label_return:								\
	return (ret);							\
}

CTL_RO_NL_GEN(version, JEMALLOC_VERSION, const char *)

static int
epoch_ctl(const size_t *mib, size_t miblen, void *oldp, size_t *oldlenp,
    void *newp, size_t newlen)
{
	int ret;
	uint64_t newval;

	malloc_mutex_lock(&ctl_mtx);
	WRITE(newval, uint64_t);
	if (newp != NULL)
		ctl_refresh();
	READ(ctl_epoch, uint64_t);

	ret = 0;
label_return:
	malloc_mutex_unlock(&ctl_mtx);
	return (ret);
}

static int
thread_tcache_enabled_ctl(const size_t *mib, size_t miblen, void *oldp,
    size_t *oldlenp, void *newp, size_t newlen)
{
	int ret;
	bool oldval;

	if (config_tcache == false)
		return (ENOENT);

	oldval = tcache_enabled_get();
	if (newp != NULL) {
		if (newlen != sizeof(bool)) {
			ret = EINVAL;
			goto label_return;
		}
		tcache_enabled_set(*(bool *)newp);
	}
	READ(oldval, bool);

	ret = 0;
label_return:
	return (ret);
}

static int
thread_tcache_flush_ctl(const size_t *mib, size_t miblen, void *oldp,
    size_t *oldlenp, void *newp, size_t newlen)
{
	int ret;

	if (config_tcache == false)
		return (ENOENT);

	READONLY();
	WRITEONLY();

	tcache_flush();

	ret = 0;
label_return:
	return (ret);
}

static int
thread_arena_ctl(const size_t *mib, size_t miblen, void *oldp, size_t *oldlenp,
    void *newp, size_t newlen)
{
	int ret;
	unsigned newind, oldind;

	malloc_mutex_lock(&ctl_mtx);
	newind = oldind = choose_arena(NULL)->ind;
	WRITE(newind, unsigned);
	READ(oldind, unsigned);
	if (newind != oldind) {
		arena_t *arena;

		if (newind >= ctl_stats.narenas) {
			/* New arena index is out of range. */
			ret = EFAULT;
			goto label_return;
		}

		/* Initialize arena if necessary. */
		malloc_mutex_lock(&arenas_lock);
		if ((arena = arenas[newind]) == NULL && (arena =
		    arenas_extend(newind)) == NULL) {
			malloc_mutex_unlock(&arenas_lock);
			ret = EAGAIN;
			goto label_return;
		}
		assert(arena == arenas[newind]);
		arenas[oldind]->nthreads--;
		arenas[newind]->nthreads++;
		malloc_mutex_unlock(&arenas_lock);

		/* Set new arena association. */
		if (config_tcache) {
			tcache_t *tcache;
			if ((uintptr_t)(tcache = *tcache_tsd_get()) >
			    (uintptr_t)TCACHE_STATE_MAX) {
				tcache_arena_dissociate(tcache);
				tcache_arena_associate(tcache, arena);
			}
		}
		arenas_tsd_set(&arena);
	}

	ret = 0;
label_return:
	malloc_mutex_unlock(&ctl_mtx);
	return (ret);
}

CTL_RO_NL_CGEN(config_stats, thread_allocated,
    thread_allocated_tsd_get()->allocated, uint64_t)
CTL_RO_NL_CGEN(config_stats, thread_allocatedp,
    &thread_allocated_tsd_get()->allocated, uint64_t *)
CTL_RO_NL_CGEN(config_stats, thread_deallocated,
    thread_allocated_tsd_get()->deallocated, uint64_t)
CTL_RO_NL_CGEN(config_stats, thread_deallocatedp,
    &thread_allocated_tsd_get()->deallocated, uint64_t *)

/******************************************************************************/

CTL_RO_BOOL_CONFIG_GEN(config_debug)
CTL_RO_BOOL_CONFIG_GEN(config_dss)
CTL_RO_BOOL_CONFIG_GEN(config_fill)
CTL_RO_BOOL_CONFIG_GEN(config_lazy_lock)
CTL_RO_BOOL_CONFIG_GEN(config_mremap)
CTL_RO_BOOL_CONFIG_GEN(config_munmap)
CTL_RO_BOOL_CONFIG_GEN(config_prof)
CTL_RO_BOOL_CONFIG_GEN(config_prof_libgcc)
CTL_RO_BOOL_CONFIG_GEN(config_prof_libunwind)
CTL_RO_BOOL_CONFIG_GEN(config_stats)
CTL_RO_BOOL_CONFIG_GEN(config_tcache)
CTL_RO_BOOL_CONFIG_GEN(config_tls)
CTL_RO_BOOL_CONFIG_GEN(config_utrace)
CTL_RO_BOOL_CONFIG_GEN(config_valgrind)
CTL_RO_BOOL_CONFIG_GEN(config_xmalloc)

/******************************************************************************/

CTL_RO_NL_GEN(opt_abort, opt_abort, bool)
CTL_RO_NL_GEN(opt_dss, opt_dss, const char *)
CTL_RO_NL_GEN(opt_lg_chunk, opt_lg_chunk, size_t)
CTL_RO_NL_GEN(opt_narenas, opt_narenas, size_t)
CTL_RO_NL_GEN(opt_lg_dirty_mult, opt_lg_dirty_mult, ssize_t)
CTL_RO_NL_GEN(opt_stats_print, opt_stats_print, bool)
CTL_RO_NL_CGEN(config_fill, opt_junk, opt_junk, bool)
CTL_RO_NL_CGEN(config_fill, opt_zero, opt_zero, bool)
CTL_RO_NL_CGEN(config_fill, opt_quarantine, opt_quarantine, size_t)
CTL_RO_NL_CGEN(config_fill, opt_redzone, opt_redzone, bool)
CTL_RO_NL_CGEN(config_utrace, opt_utrace, opt_utrace, bool)
CTL_RO_NL_CGEN(config_valgrind, opt_valgrind, opt_valgrind, bool)
CTL_RO_NL_CGEN(config_xmalloc, opt_xmalloc, opt_xmalloc, bool)
CTL_RO_NL_CGEN(config_tcache, opt_tcache, opt_tcache, bool)
CTL_RO_NL_CGEN(config_tcache, opt_lg_tcache_max, opt_lg_tcache_max, ssize_t)
CTL_RO_NL_CGEN(config_prof, opt_prof, opt_prof, bool)
CTL_RO_NL_CGEN(config_prof, opt_prof_prefix, opt_prof_prefix, const char *)
CTL_RO_CGEN(config_prof, opt_prof_active, opt_prof_active, bool) /* Mutable. */
CTL_RO_NL_CGEN(config_prof, opt_lg_prof_sample, opt_lg_prof_sample, size_t)
CTL_RO_NL_CGEN(config_prof, opt_lg_prof_interval, opt_lg_prof_interval, ssize_t)
CTL_RO_NL_CGEN(config_prof, opt_prof_gdump, opt_prof_gdump, bool)
CTL_RO_NL_CGEN(config_prof, opt_prof_final, opt_prof_final, bool)
CTL_RO_NL_CGEN(config_prof, opt_prof_leak, opt_prof_leak, bool)
CTL_RO_NL_CGEN(config_prof, opt_prof_accum, opt_prof_accum, bool)

/******************************************************************************/

/* ctl_mutex must be held during execution of this function. */
static void
arena_purge(unsigned arena_ind)
{
	VARIABLE_ARRAY(arena_t *, tarenas, ctl_stats.narenas);

	malloc_mutex_lock(&arenas_lock);
	memcpy(tarenas, arenas, sizeof(arena_t *) * ctl_stats.narenas);
	malloc_mutex_unlock(&arenas_lock);

	if (arena_ind == ctl_stats.narenas) {
		unsigned i;
		for (i = 0; i < ctl_stats.narenas; i++) {
			if (tarenas[i] != NULL)
				arena_purge_all(tarenas[i]);
		}
	} else {
		assert(arena_ind < ctl_stats.narenas);
		if (tarenas[arena_ind] != NULL)
			arena_purge_all(tarenas[arena_ind]);
	}
}

static int
arena_i_purge_ctl(const size_t *mib, size_t miblen, void *oldp, size_t *oldlenp,
    void *newp, size_t newlen)
{
	int ret;

	READONLY();
	WRITEONLY();
	malloc_mutex_lock(&ctl_mtx);
	arena_purge(mib[1]);
	malloc_mutex_unlock(&ctl_mtx);

	ret = 0;
label_return:
	return (ret);
}

static int
arena_i_dss_ctl(const size_t *mib, size_t miblen, void *oldp, size_t *oldlenp,
    void *newp, size_t newlen)
{
	int ret, i;
	bool match, err;
	const char *dss;
	unsigned arena_ind = mib[1];
	dss_prec_t dss_prec_old = dss_prec_limit;
	dss_prec_t dss_prec = dss_prec_limit;

	malloc_mutex_lock(&ctl_mtx);
	WRITE(dss, const char *);
	match = false;
	for (i = 0; i < dss_prec_limit; i++) {
		if (strcmp(dss_prec_names[i], dss) == 0) {
			dss_prec = i;
			match = true;
			break;
		}
	}
	if (match == false) {
		ret = EINVAL;
		goto label_return;
	}

	if (arena_ind < ctl_stats.narenas) {
		arena_t *arena = arenas[arena_ind];
		if (arena != NULL) {
			dss_prec_old = arena_dss_prec_get(arena);
			arena_dss_prec_set(arena, dss_prec);
			err = false;
		} else
			err = true;
	} else {
		dss_prec_old = chunk_dss_prec_get();
		err = chunk_dss_prec_set(dss_prec);
	}
	dss = dss_prec_names[dss_prec_old];
	READ(dss, const char *);
	if (err) {
		ret = EFAULT;
		goto label_return;
	}

	ret = 0;
label_return:
	malloc_mutex_unlock(&ctl_mtx);
	return (ret);
}

static const ctl_named_node_t *
arena_i_index(const size_t *mib, size_t miblen, size_t i)
{
	const ctl_named_node_t * ret;

	malloc_mutex_lock(&ctl_mtx);
	if (i > ctl_stats.narenas) {
		ret = NULL;
		goto label_return;
	}

	ret = super_arena_i_node;
label_return:
	malloc_mutex_unlock(&ctl_mtx);
	return (ret);
}


/******************************************************************************/

CTL_RO_NL_GEN(arenas_bin_i_size, arena_bin_info[mib[2]].reg_size, size_t)
CTL_RO_NL_GEN(arenas_bin_i_nregs, arena_bin_info[mib[2]].nregs, uint32_t)
CTL_RO_NL_GEN(arenas_bin_i_run_size, arena_bin_info[mib[2]].run_size, size_t)
static const ctl_named_node_t *
arenas_bin_i_index(const size_t *mib, size_t miblen, size_t i)
{

	if (i > NBINS)
		return (NULL);
	return (super_arenas_bin_i_node);
}

CTL_RO_NL_GEN(arenas_lrun_i_size, ((mib[2]+1) << LG_PAGE), size_t)
static const ctl_named_node_t *
arenas_lrun_i_index(const size_t *mib, size_t miblen, size_t i)
{

	if (i > nlclasses)
		return (NULL);
	return (super_arenas_lrun_i_node);
}

static int
arenas_narenas_ctl(const size_t *mib, size_t miblen, void *oldp,
    size_t *oldlenp, void *newp, size_t newlen)
{
	int ret;
	unsigned narenas;

	malloc_mutex_lock(&ctl_mtx);
	READONLY();
	if (*oldlenp != sizeof(unsigned)) {
		ret = EINVAL;
		goto label_return;
	}
	narenas = ctl_stats.narenas;
	READ(narenas, unsigned);

	ret = 0;
label_return:
	malloc_mutex_unlock(&ctl_mtx);
	return (ret);
}

static int
arenas_initialized_ctl(const size_t *mib, size_t miblen, void *oldp,
    size_t *oldlenp, void *newp, size_t newlen)
{
	int ret;
	unsigned nread, i;

	malloc_mutex_lock(&ctl_mtx);
	READONLY();
	if (*oldlenp != ctl_stats.narenas * sizeof(bool)) {
		ret = EINVAL;
		nread = (*oldlenp < ctl_stats.narenas * sizeof(bool))
		    ? (*oldlenp / sizeof(bool)) : ctl_stats.narenas;
	} else {
		ret = 0;
		nread = ctl_stats.narenas;
	}

	for (i = 0; i < nread; i++)
		((bool *)oldp)[i] = ctl_stats.arenas[i].initialized;

label_return:
	malloc_mutex_unlock(&ctl_mtx);
	return (ret);
}

CTL_RO_NL_GEN(arenas_quantum, QUANTUM, size_t)
CTL_RO_NL_GEN(arenas_page, PAGE, size_t)
CTL_RO_NL_CGEN(config_tcache, arenas_tcache_max, tcache_maxclass, size_t)
CTL_RO_NL_GEN(arenas_nbins, NBINS, unsigned)
CTL_RO_NL_CGEN(config_tcache, arenas_nhbins, nhbins, unsigned)
CTL_RO_NL_GEN(arenas_nlruns, nlclasses, size_t)

static int
arenas_purge_ctl(const size_t *mib, size_t miblen, void *oldp, size_t *oldlenp,
    void *newp, size_t newlen)
{
	int ret;
	unsigned arena_ind;

	malloc_mutex_lock(&ctl_mtx);
	WRITEONLY();
	arena_ind = UINT_MAX;
	WRITE(arena_ind, unsigned);
	if (newp != NULL && arena_ind >= ctl_stats.narenas)
		ret = EFAULT;
	else {
		if (arena_ind == UINT_MAX)
			arena_ind = ctl_stats.narenas;
		arena_purge(arena_ind);
		ret = 0;
	}

label_return:
	malloc_mutex_unlock(&ctl_mtx);
	return (ret);
}

static int
arenas_extend_ctl(const size_t *mib, size_t miblen, void *oldp, size_t *oldlenp,
    void *newp, size_t newlen)
{
	int ret;
	unsigned narenas;

	malloc_mutex_lock(&ctl_mtx);
	READONLY();
	if (ctl_grow()) {
		ret = EAGAIN;
		goto label_return;
	}
	narenas = ctl_stats.narenas - 1;
	READ(narenas, unsigned);

	ret = 0;
label_return:
	malloc_mutex_unlock(&ctl_mtx);
	return (ret);
}

/******************************************************************************/

static int
prof_active_ctl(const size_t *mib, size_t miblen, void *oldp, size_t *oldlenp,
    void *newp, size_t newlen)
{
	int ret;
	bool oldval;

	if (config_prof == false)
		return (ENOENT);

	malloc_mutex_lock(&ctl_mtx); /* Protect opt_prof_active. */
	oldval = opt_prof_active;
	if (newp != NULL) {
		/*
		 * The memory barriers will tend to make opt_prof_active
		 * propagate faster on systems with weak memory ordering.
		 */
		mb_write();
		WRITE(opt_prof_active, bool);
		mb_write();
	}
	READ(oldval, bool);

	ret = 0;
label_return:
	malloc_mutex_unlock(&ctl_mtx);
	return (ret);
}

static int
prof_dump_ctl(const size_t *mib, size_t miblen, void *oldp, size_t *oldlenp,
    void *newp, size_t newlen)
{
	int ret;
	const char *filename = NULL;

	if (config_prof == false)
		return (ENOENT);

	WRITEONLY();
	WRITE(filename, const char *);

	if (prof_mdump(filename)) {
		ret = EFAULT;
		goto label_return;
	}

	ret = 0;
label_return:
	return (ret);
}

CTL_RO_NL_CGEN(config_prof, prof_interval, prof_interval, uint64_t)

/******************************************************************************/

CTL_RO_CGEN(config_stats, stats_chunks_current, ctl_stats.chunks.current,
    size_t)
CTL_RO_CGEN(config_stats, stats_chunks_total, ctl_stats.chunks.total, uint64_t)
CTL_RO_CGEN(config_stats, stats_chunks_high, ctl_stats.chunks.high, size_t)
CTL_RO_CGEN(config_stats, stats_huge_allocated, huge_allocated, size_t)
CTL_RO_CGEN(config_stats, stats_huge_nmalloc, huge_nmalloc, uint64_t)
CTL_RO_CGEN(config_stats, stats_huge_ndalloc, huge_ndalloc, uint64_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_small_allocated,
    ctl_stats.arenas[mib[2]].allocated_small, size_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_small_nmalloc,
    ctl_stats.arenas[mib[2]].nmalloc_small, uint64_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_small_ndalloc,
    ctl_stats.arenas[mib[2]].ndalloc_small, uint64_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_small_nrequests,
    ctl_stats.arenas[mib[2]].nrequests_small, uint64_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_large_allocated,
    ctl_stats.arenas[mib[2]].astats.allocated_large, size_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_large_nmalloc,
    ctl_stats.arenas[mib[2]].astats.nmalloc_large, uint64_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_large_ndalloc,
    ctl_stats.arenas[mib[2]].astats.ndalloc_large, uint64_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_large_nrequests,
    ctl_stats.arenas[mib[2]].astats.nrequests_large, uint64_t)

CTL_RO_CGEN(config_stats, stats_arenas_i_bins_j_allocated,
    ctl_stats.arenas[mib[2]].bstats[mib[4]].allocated, size_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_bins_j_nmalloc,
    ctl_stats.arenas[mib[2]].bstats[mib[4]].nmalloc, uint64_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_bins_j_ndalloc,
    ctl_stats.arenas[mib[2]].bstats[mib[4]].ndalloc, uint64_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_bins_j_nrequests,
    ctl_stats.arenas[mib[2]].bstats[mib[4]].nrequests, uint64_t)
CTL_RO_CGEN(config_stats && config_tcache, stats_arenas_i_bins_j_nfills,
    ctl_stats.arenas[mib[2]].bstats[mib[4]].nfills, uint64_t)
CTL_RO_CGEN(config_stats && config_tcache, stats_arenas_i_bins_j_nflushes,
    ctl_stats.arenas[mib[2]].bstats[mib[4]].nflushes, uint64_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_bins_j_nruns,
    ctl_stats.arenas[mib[2]].bstats[mib[4]].nruns, uint64_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_bins_j_nreruns,
    ctl_stats.arenas[mib[2]].bstats[mib[4]].reruns, uint64_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_bins_j_curruns,
    ctl_stats.arenas[mib[2]].bstats[mib[4]].curruns, size_t)

static const ctl_named_node_t *
stats_arenas_i_bins_j_index(const size_t *mib, size_t miblen, size_t j)
{

	if (j > NBINS)
		return (NULL);
	return (super_stats_arenas_i_bins_j_node);
}

CTL_RO_CGEN(config_stats, stats_arenas_i_lruns_j_nmalloc,
    ctl_stats.arenas[mib[2]].lstats[mib[4]].nmalloc, uint64_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_lruns_j_ndalloc,
    ctl_stats.arenas[mib[2]].lstats[mib[4]].ndalloc, uint64_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_lruns_j_nrequests,
    ctl_stats.arenas[mib[2]].lstats[mib[4]].nrequests, uint64_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_lruns_j_curruns,
    ctl_stats.arenas[mib[2]].lstats[mib[4]].curruns, size_t)

static const ctl_named_node_t *
stats_arenas_i_lruns_j_index(const size_t *mib, size_t miblen, size_t j)
{

	if (j > nlclasses)
		return (NULL);
	return (super_stats_arenas_i_lruns_j_node);
}

CTL_RO_GEN(stats_arenas_i_nthreads, ctl_stats.arenas[mib[2]].nthreads, unsigned)
CTL_RO_GEN(stats_arenas_i_dss, ctl_stats.arenas[mib[2]].dss, const char *)
CTL_RO_GEN(stats_arenas_i_pactive, ctl_stats.arenas[mib[2]].pactive, size_t)
CTL_RO_GEN(stats_arenas_i_pdirty, ctl_stats.arenas[mib[2]].pdirty, size_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_mapped,
    ctl_stats.arenas[mib[2]].astats.mapped, size_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_npurge,
    ctl_stats.arenas[mib[2]].astats.npurge, uint64_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_nmadvise,
    ctl_stats.arenas[mib[2]].astats.nmadvise, uint64_t)
CTL_RO_CGEN(config_stats, stats_arenas_i_purged,
    ctl_stats.arenas[mib[2]].astats.purged, uint64_t)

static const ctl_named_node_t *
stats_arenas_i_index(const size_t *mib, size_t miblen, size_t i)
{
	const ctl_named_node_t * ret;

	malloc_mutex_lock(&ctl_mtx);
	if (i > ctl_stats.narenas || ctl_stats.arenas[i].initialized == false) {
		ret = NULL;
		goto label_return;
	}

	ret = super_stats_arenas_i_node;
label_return:
	malloc_mutex_unlock(&ctl_mtx);
	return (ret);
}

CTL_RO_CGEN(config_stats, stats_cactive, &stats_cactive, size_t *)
CTL_RO_CGEN(config_stats, stats_allocated, ctl_stats.allocated, size_t)
CTL_RO_CGEN(config_stats, stats_active, ctl_stats.active, size_t)
CTL_RO_CGEN(config_stats, stats_mapped, ctl_stats.mapped, size_t)
