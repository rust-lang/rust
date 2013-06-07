#define	JEMALLOC_STATS_C_
#include "jemalloc/internal/jemalloc_internal.h"

#define	CTL_GET(n, v, t) do {						\
	size_t sz = sizeof(t);						\
	xmallctl(n, v, &sz, NULL, 0);					\
} while (0)

#define	CTL_I_GET(n, v, t) do {						\
	size_t mib[6];							\
	size_t miblen = sizeof(mib) / sizeof(size_t);			\
	size_t sz = sizeof(t);						\
	xmallctlnametomib(n, mib, &miblen);				\
	mib[2] = i;							\
	xmallctlbymib(mib, miblen, v, &sz, NULL, 0);			\
} while (0)

#define	CTL_J_GET(n, v, t) do {						\
	size_t mib[6];							\
	size_t miblen = sizeof(mib) / sizeof(size_t);			\
	size_t sz = sizeof(t);						\
	xmallctlnametomib(n, mib, &miblen);				\
	mib[2] = j;							\
	xmallctlbymib(mib, miblen, v, &sz, NULL, 0);			\
} while (0)

#define	CTL_IJ_GET(n, v, t) do {					\
	size_t mib[6];							\
	size_t miblen = sizeof(mib) / sizeof(size_t);			\
	size_t sz = sizeof(t);						\
	xmallctlnametomib(n, mib, &miblen);				\
	mib[2] = i;							\
	mib[4] = j;							\
	xmallctlbymib(mib, miblen, v, &sz, NULL, 0);			\
} while (0)

/******************************************************************************/
/* Data. */

bool	opt_stats_print = false;

size_t	stats_cactive = 0;

/******************************************************************************/
/* Function prototypes for non-inline static functions. */

static void	stats_arena_bins_print(void (*write_cb)(void *, const char *),
    void *cbopaque, unsigned i);
static void	stats_arena_lruns_print(void (*write_cb)(void *, const char *),
    void *cbopaque, unsigned i);
static void	stats_arena_print(void (*write_cb)(void *, const char *),
    void *cbopaque, unsigned i, bool bins, bool large);

/******************************************************************************/

static void
stats_arena_bins_print(void (*write_cb)(void *, const char *), void *cbopaque,
    unsigned i)
{
	size_t page;
	bool config_tcache;
	unsigned nbins, j, gap_start;

	CTL_GET("arenas.page", &page, size_t);

	CTL_GET("config.tcache", &config_tcache, bool);
	if (config_tcache) {
		malloc_cprintf(write_cb, cbopaque,
		    "bins:     bin  size regs pgs    allocated      nmalloc"
		    "      ndalloc    nrequests       nfills     nflushes"
		    "      newruns       reruns      curruns\n");
	} else {
		malloc_cprintf(write_cb, cbopaque,
		    "bins:     bin  size regs pgs    allocated      nmalloc"
		    "      ndalloc      newruns       reruns      curruns\n");
	}
	CTL_GET("arenas.nbins", &nbins, unsigned);
	for (j = 0, gap_start = UINT_MAX; j < nbins; j++) {
		uint64_t nruns;

		CTL_IJ_GET("stats.arenas.0.bins.0.nruns", &nruns, uint64_t);
		if (nruns == 0) {
			if (gap_start == UINT_MAX)
				gap_start = j;
		} else {
			size_t reg_size, run_size, allocated;
			uint32_t nregs;
			uint64_t nmalloc, ndalloc, nrequests, nfills, nflushes;
			uint64_t reruns;
			size_t curruns;

			if (gap_start != UINT_MAX) {
				if (j > gap_start + 1) {
					/* Gap of more than one size class. */
					malloc_cprintf(write_cb, cbopaque,
					    "[%u..%u]\n", gap_start,
					    j - 1);
				} else {
					/* Gap of one size class. */
					malloc_cprintf(write_cb, cbopaque,
					    "[%u]\n", gap_start);
				}
				gap_start = UINT_MAX;
			}
			CTL_J_GET("arenas.bin.0.size", &reg_size, size_t);
			CTL_J_GET("arenas.bin.0.nregs", &nregs, uint32_t);
			CTL_J_GET("arenas.bin.0.run_size", &run_size, size_t);
			CTL_IJ_GET("stats.arenas.0.bins.0.allocated",
			    &allocated, size_t);
			CTL_IJ_GET("stats.arenas.0.bins.0.nmalloc",
			    &nmalloc, uint64_t);
			CTL_IJ_GET("stats.arenas.0.bins.0.ndalloc",
			    &ndalloc, uint64_t);
			if (config_tcache) {
				CTL_IJ_GET("stats.arenas.0.bins.0.nrequests",
				    &nrequests, uint64_t);
				CTL_IJ_GET("stats.arenas.0.bins.0.nfills",
				    &nfills, uint64_t);
				CTL_IJ_GET("stats.arenas.0.bins.0.nflushes",
				    &nflushes, uint64_t);
			}
			CTL_IJ_GET("stats.arenas.0.bins.0.nreruns", &reruns,
			    uint64_t);
			CTL_IJ_GET("stats.arenas.0.bins.0.curruns", &curruns,
			    size_t);
			if (config_tcache) {
				malloc_cprintf(write_cb, cbopaque,
				    "%13u %5zu %4u %3zu %12zu %12"PRIu64
				    " %12"PRIu64" %12"PRIu64" %12"PRIu64
				    " %12"PRIu64" %12"PRIu64" %12"PRIu64
				    " %12zu\n",
				    j, reg_size, nregs, run_size / page,
				    allocated, nmalloc, ndalloc, nrequests,
				    nfills, nflushes, nruns, reruns, curruns);
			} else {
				malloc_cprintf(write_cb, cbopaque,
				    "%13u %5zu %4u %3zu %12zu %12"PRIu64
				    " %12"PRIu64" %12"PRIu64" %12"PRIu64
				    " %12zu\n",
				    j, reg_size, nregs, run_size / page,
				    allocated, nmalloc, ndalloc, nruns, reruns,
				    curruns);
			}
		}
	}
	if (gap_start != UINT_MAX) {
		if (j > gap_start + 1) {
			/* Gap of more than one size class. */
			malloc_cprintf(write_cb, cbopaque, "[%u..%u]\n",
			    gap_start, j - 1);
		} else {
			/* Gap of one size class. */
			malloc_cprintf(write_cb, cbopaque, "[%u]\n", gap_start);
		}
	}
}

static void
stats_arena_lruns_print(void (*write_cb)(void *, const char *), void *cbopaque,
    unsigned i)
{
	size_t page, nlruns, j;
	ssize_t gap_start;

	CTL_GET("arenas.page", &page, size_t);

	malloc_cprintf(write_cb, cbopaque,
	    "large:   size pages      nmalloc      ndalloc    nrequests"
	    "      curruns\n");
	CTL_GET("arenas.nlruns", &nlruns, size_t);
	for (j = 0, gap_start = -1; j < nlruns; j++) {
		uint64_t nmalloc, ndalloc, nrequests;
		size_t run_size, curruns;

		CTL_IJ_GET("stats.arenas.0.lruns.0.nmalloc", &nmalloc,
		    uint64_t);
		CTL_IJ_GET("stats.arenas.0.lruns.0.ndalloc", &ndalloc,
		    uint64_t);
		CTL_IJ_GET("stats.arenas.0.lruns.0.nrequests", &nrequests,
		    uint64_t);
		if (nrequests == 0) {
			if (gap_start == -1)
				gap_start = j;
		} else {
			CTL_J_GET("arenas.lrun.0.size", &run_size, size_t);
			CTL_IJ_GET("stats.arenas.0.lruns.0.curruns", &curruns,
			    size_t);
			if (gap_start != -1) {
				malloc_cprintf(write_cb, cbopaque, "[%zu]\n",
				    j - gap_start);
				gap_start = -1;
			}
			malloc_cprintf(write_cb, cbopaque,
			    "%13zu %5zu %12"PRIu64" %12"PRIu64" %12"PRIu64
			    " %12zu\n",
			    run_size, run_size / page, nmalloc, ndalloc,
			    nrequests, curruns);
		}
	}
	if (gap_start != -1)
		malloc_cprintf(write_cb, cbopaque, "[%zu]\n", j - gap_start);
}

static void
stats_arena_print(void (*write_cb)(void *, const char *), void *cbopaque,
    unsigned i, bool bins, bool large)
{
	unsigned nthreads;
	const char *dss;
	size_t page, pactive, pdirty, mapped;
	uint64_t npurge, nmadvise, purged;
	size_t small_allocated;
	uint64_t small_nmalloc, small_ndalloc, small_nrequests;
	size_t large_allocated;
	uint64_t large_nmalloc, large_ndalloc, large_nrequests;

	CTL_GET("arenas.page", &page, size_t);

	CTL_I_GET("stats.arenas.0.nthreads", &nthreads, unsigned);
	malloc_cprintf(write_cb, cbopaque,
	    "assigned threads: %u\n", nthreads);
	CTL_I_GET("stats.arenas.0.dss", &dss, const char *);
	malloc_cprintf(write_cb, cbopaque, "dss allocation precedence: %s\n",
	    dss);
	CTL_I_GET("stats.arenas.0.pactive", &pactive, size_t);
	CTL_I_GET("stats.arenas.0.pdirty", &pdirty, size_t);
	CTL_I_GET("stats.arenas.0.npurge", &npurge, uint64_t);
	CTL_I_GET("stats.arenas.0.nmadvise", &nmadvise, uint64_t);
	CTL_I_GET("stats.arenas.0.purged", &purged, uint64_t);
	malloc_cprintf(write_cb, cbopaque,
	    "dirty pages: %zu:%zu active:dirty, %"PRIu64" sweep%s,"
	    " %"PRIu64" madvise%s, %"PRIu64" purged\n",
	    pactive, pdirty, npurge, npurge == 1 ? "" : "s",
	    nmadvise, nmadvise == 1 ? "" : "s", purged);

	malloc_cprintf(write_cb, cbopaque,
	    "            allocated      nmalloc      ndalloc    nrequests\n");
	CTL_I_GET("stats.arenas.0.small.allocated", &small_allocated, size_t);
	CTL_I_GET("stats.arenas.0.small.nmalloc", &small_nmalloc, uint64_t);
	CTL_I_GET("stats.arenas.0.small.ndalloc", &small_ndalloc, uint64_t);
	CTL_I_GET("stats.arenas.0.small.nrequests", &small_nrequests, uint64_t);
	malloc_cprintf(write_cb, cbopaque,
	    "small:   %12zu %12"PRIu64" %12"PRIu64" %12"PRIu64"\n",
	    small_allocated, small_nmalloc, small_ndalloc, small_nrequests);
	CTL_I_GET("stats.arenas.0.large.allocated", &large_allocated, size_t);
	CTL_I_GET("stats.arenas.0.large.nmalloc", &large_nmalloc, uint64_t);
	CTL_I_GET("stats.arenas.0.large.ndalloc", &large_ndalloc, uint64_t);
	CTL_I_GET("stats.arenas.0.large.nrequests", &large_nrequests, uint64_t);
	malloc_cprintf(write_cb, cbopaque,
	    "large:   %12zu %12"PRIu64" %12"PRIu64" %12"PRIu64"\n",
	    large_allocated, large_nmalloc, large_ndalloc, large_nrequests);
	malloc_cprintf(write_cb, cbopaque,
	    "total:   %12zu %12"PRIu64" %12"PRIu64" %12"PRIu64"\n",
	    small_allocated + large_allocated,
	    small_nmalloc + large_nmalloc,
	    small_ndalloc + large_ndalloc,
	    small_nrequests + large_nrequests);
	malloc_cprintf(write_cb, cbopaque, "active:  %12zu\n", pactive * page);
	CTL_I_GET("stats.arenas.0.mapped", &mapped, size_t);
	malloc_cprintf(write_cb, cbopaque, "mapped:  %12zu\n", mapped);

	if (bins)
		stats_arena_bins_print(write_cb, cbopaque, i);
	if (large)
		stats_arena_lruns_print(write_cb, cbopaque, i);
}

void
stats_print(void (*write_cb)(void *, const char *), void *cbopaque,
    const char *opts)
{
	int err;
	uint64_t epoch;
	size_t u64sz;
	bool general = true;
	bool merged = true;
	bool unmerged = true;
	bool bins = true;
	bool large = true;

	/*
	 * Refresh stats, in case mallctl() was called by the application.
	 *
	 * Check for OOM here, since refreshing the ctl cache can trigger
	 * allocation.  In practice, none of the subsequent mallctl()-related
	 * calls in this function will cause OOM if this one succeeds.
	 * */
	epoch = 1;
	u64sz = sizeof(uint64_t);
	err = je_mallctl("epoch", &epoch, &u64sz, &epoch, sizeof(uint64_t));
	if (err != 0) {
		if (err == EAGAIN) {
			malloc_write("<jemalloc>: Memory allocation failure in "
			    "mallctl(\"epoch\", ...)\n");
			return;
		}
		malloc_write("<jemalloc>: Failure in mallctl(\"epoch\", "
		    "...)\n");
		abort();
	}

	if (opts != NULL) {
		unsigned i;

		for (i = 0; opts[i] != '\0'; i++) {
			switch (opts[i]) {
			case 'g':
				general = false;
				break;
			case 'm':
				merged = false;
				break;
			case 'a':
				unmerged = false;
				break;
			case 'b':
				bins = false;
				break;
			case 'l':
				large = false;
				break;
			default:;
			}
		}
	}

	malloc_cprintf(write_cb, cbopaque,
	    "___ Begin jemalloc statistics ___\n");
	if (general) {
		int err;
		const char *cpv;
		bool bv;
		unsigned uv;
		ssize_t ssv;
		size_t sv, bsz, ssz, sssz, cpsz;

		bsz = sizeof(bool);
		ssz = sizeof(size_t);
		sssz = sizeof(ssize_t);
		cpsz = sizeof(const char *);

		CTL_GET("version", &cpv, const char *);
		malloc_cprintf(write_cb, cbopaque, "Version: %s\n", cpv);
		CTL_GET("config.debug", &bv, bool);
		malloc_cprintf(write_cb, cbopaque, "Assertions %s\n",
		    bv ? "enabled" : "disabled");

#define OPT_WRITE_BOOL(n)						\
		if ((err = je_mallctl("opt."#n, &bv, &bsz, NULL, 0))	\
		    == 0) {						\
			malloc_cprintf(write_cb, cbopaque,		\
			    "  opt."#n": %s\n", bv ? "true" : "false");	\
		}
#define OPT_WRITE_SIZE_T(n)						\
		if ((err = je_mallctl("opt."#n, &sv, &ssz, NULL, 0))	\
		    == 0) {						\
			malloc_cprintf(write_cb, cbopaque,		\
			"  opt."#n": %zu\n", sv);			\
		}
#define OPT_WRITE_SSIZE_T(n)						\
		if ((err = je_mallctl("opt."#n, &ssv, &sssz, NULL, 0))	\
		    == 0) {						\
			malloc_cprintf(write_cb, cbopaque,		\
			    "  opt."#n": %zd\n", ssv);			\
		}
#define OPT_WRITE_CHAR_P(n)						\
		if ((err = je_mallctl("opt."#n, &cpv, &cpsz, NULL, 0))	\
		    == 0) {						\
			malloc_cprintf(write_cb, cbopaque,		\
			    "  opt."#n": \"%s\"\n", cpv);		\
		}

		malloc_cprintf(write_cb, cbopaque,
		    "Run-time option settings:\n");
		OPT_WRITE_BOOL(abort)
		OPT_WRITE_SIZE_T(lg_chunk)
		OPT_WRITE_CHAR_P(dss)
		OPT_WRITE_SIZE_T(narenas)
		OPT_WRITE_SSIZE_T(lg_dirty_mult)
		OPT_WRITE_BOOL(stats_print)
		OPT_WRITE_BOOL(junk)
		OPT_WRITE_SIZE_T(quarantine)
		OPT_WRITE_BOOL(redzone)
		OPT_WRITE_BOOL(zero)
		OPT_WRITE_BOOL(utrace)
		OPT_WRITE_BOOL(valgrind)
		OPT_WRITE_BOOL(xmalloc)
		OPT_WRITE_BOOL(tcache)
		OPT_WRITE_SSIZE_T(lg_tcache_max)
		OPT_WRITE_BOOL(prof)
		OPT_WRITE_CHAR_P(prof_prefix)
		OPT_WRITE_BOOL(prof_active)
		OPT_WRITE_SSIZE_T(lg_prof_sample)
		OPT_WRITE_BOOL(prof_accum)
		OPT_WRITE_SSIZE_T(lg_prof_interval)
		OPT_WRITE_BOOL(prof_gdump)
		OPT_WRITE_BOOL(prof_final)
		OPT_WRITE_BOOL(prof_leak)

#undef OPT_WRITE_BOOL
#undef OPT_WRITE_SIZE_T
#undef OPT_WRITE_SSIZE_T
#undef OPT_WRITE_CHAR_P

		malloc_cprintf(write_cb, cbopaque, "CPUs: %u\n", ncpus);

		CTL_GET("arenas.narenas", &uv, unsigned);
		malloc_cprintf(write_cb, cbopaque, "Arenas: %u\n", uv);

		malloc_cprintf(write_cb, cbopaque, "Pointer size: %zu\n",
		    sizeof(void *));

		CTL_GET("arenas.quantum", &sv, size_t);
		malloc_cprintf(write_cb, cbopaque, "Quantum size: %zu\n", sv);

		CTL_GET("arenas.page", &sv, size_t);
		malloc_cprintf(write_cb, cbopaque, "Page size: %zu\n", sv);

		CTL_GET("opt.lg_dirty_mult", &ssv, ssize_t);
		if (ssv >= 0) {
			malloc_cprintf(write_cb, cbopaque,
			    "Min active:dirty page ratio per arena: %u:1\n",
			    (1U << ssv));
		} else {
			malloc_cprintf(write_cb, cbopaque,
			    "Min active:dirty page ratio per arena: N/A\n");
		}
		if ((err = je_mallctl("arenas.tcache_max", &sv, &ssz, NULL, 0))
		    == 0) {
			malloc_cprintf(write_cb, cbopaque,
			    "Maximum thread-cached size class: %zu\n", sv);
		}
		if ((err = je_mallctl("opt.prof", &bv, &bsz, NULL, 0)) == 0 &&
		    bv) {
			CTL_GET("opt.lg_prof_sample", &sv, size_t);
			malloc_cprintf(write_cb, cbopaque,
			    "Average profile sample interval: %"PRIu64
			    " (2^%zu)\n", (((uint64_t)1U) << sv), sv);

			CTL_GET("opt.lg_prof_interval", &ssv, ssize_t);
			if (ssv >= 0) {
				malloc_cprintf(write_cb, cbopaque,
				    "Average profile dump interval: %"PRIu64
				    " (2^%zd)\n",
				    (((uint64_t)1U) << ssv), ssv);
			} else {
				malloc_cprintf(write_cb, cbopaque,
				    "Average profile dump interval: N/A\n");
			}
		}
		CTL_GET("opt.lg_chunk", &sv, size_t);
		malloc_cprintf(write_cb, cbopaque, "Chunk size: %zu (2^%zu)\n",
		    (ZU(1) << sv), sv);
	}

	if (config_stats) {
		size_t *cactive;
		size_t allocated, active, mapped;
		size_t chunks_current, chunks_high;
		uint64_t chunks_total;
		size_t huge_allocated;
		uint64_t huge_nmalloc, huge_ndalloc;

		CTL_GET("stats.cactive", &cactive, size_t *);
		CTL_GET("stats.allocated", &allocated, size_t);
		CTL_GET("stats.active", &active, size_t);
		CTL_GET("stats.mapped", &mapped, size_t);
		malloc_cprintf(write_cb, cbopaque,
		    "Allocated: %zu, active: %zu, mapped: %zu\n",
		    allocated, active, mapped);
		malloc_cprintf(write_cb, cbopaque,
		    "Current active ceiling: %zu\n", atomic_read_z(cactive));

		/* Print chunk stats. */
		CTL_GET("stats.chunks.total", &chunks_total, uint64_t);
		CTL_GET("stats.chunks.high", &chunks_high, size_t);
		CTL_GET("stats.chunks.current", &chunks_current, size_t);
		malloc_cprintf(write_cb, cbopaque, "chunks: nchunks   "
		    "highchunks    curchunks\n");
		malloc_cprintf(write_cb, cbopaque,
		    "  %13"PRIu64" %12zu %12zu\n",
		    chunks_total, chunks_high, chunks_current);

		/* Print huge stats. */
		CTL_GET("stats.huge.nmalloc", &huge_nmalloc, uint64_t);
		CTL_GET("stats.huge.ndalloc", &huge_ndalloc, uint64_t);
		CTL_GET("stats.huge.allocated", &huge_allocated, size_t);
		malloc_cprintf(write_cb, cbopaque,
		    "huge: nmalloc      ndalloc    allocated\n");
		malloc_cprintf(write_cb, cbopaque,
		    " %12"PRIu64" %12"PRIu64" %12zu\n",
		    huge_nmalloc, huge_ndalloc, huge_allocated);

		if (merged) {
			unsigned narenas;

			CTL_GET("arenas.narenas", &narenas, unsigned);
			{
				VARIABLE_ARRAY(bool, initialized, narenas);
				size_t isz;
				unsigned i, ninitialized;

				isz = sizeof(bool) * narenas;
				xmallctl("arenas.initialized", initialized,
				    &isz, NULL, 0);
				for (i = ninitialized = 0; i < narenas; i++) {
					if (initialized[i])
						ninitialized++;
				}

				if (ninitialized > 1 || unmerged == false) {
					/* Print merged arena stats. */
					malloc_cprintf(write_cb, cbopaque,
					    "\nMerged arenas stats:\n");
					stats_arena_print(write_cb, cbopaque,
					    narenas, bins, large);
				}
			}
		}

		if (unmerged) {
			unsigned narenas;

			/* Print stats for each arena. */

			CTL_GET("arenas.narenas", &narenas, unsigned);
			{
				VARIABLE_ARRAY(bool, initialized, narenas);
				size_t isz;
				unsigned i;

				isz = sizeof(bool) * narenas;
				xmallctl("arenas.initialized", initialized,
				    &isz, NULL, 0);

				for (i = 0; i < narenas; i++) {
					if (initialized[i]) {
						malloc_cprintf(write_cb,
						    cbopaque,
						    "\narenas[%u]:\n", i);
						stats_arena_print(write_cb,
						    cbopaque, i, bins, large);
					}
				}
			}
		}
	}
	malloc_cprintf(write_cb, cbopaque, "--- End jemalloc statistics ---\n");
}
