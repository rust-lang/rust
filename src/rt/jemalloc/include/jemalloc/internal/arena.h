/******************************************************************************/
#ifdef JEMALLOC_H_TYPES

/*
 * RUN_MAX_OVRHD indicates maximum desired run header overhead.  Runs are sized
 * as small as possible such that this setting is still honored, without
 * violating other constraints.  The goal is to make runs as small as possible
 * without exceeding a per run external fragmentation threshold.
 *
 * We use binary fixed point math for overhead computations, where the binary
 * point is implicitly RUN_BFP bits to the left.
 *
 * Note that it is possible to set RUN_MAX_OVRHD low enough that it cannot be
 * honored for some/all object sizes, since when heap profiling is enabled
 * there is one pointer of header overhead per object (plus a constant).  This
 * constraint is relaxed (ignored) for runs that are so small that the
 * per-region overhead is greater than:
 *
 *   (RUN_MAX_OVRHD / (reg_interval << (3+RUN_BFP))
 */
#define	RUN_BFP			12
/*                                    \/   Implicit binary fixed point. */
#define	RUN_MAX_OVRHD		0x0000003dU
#define	RUN_MAX_OVRHD_RELAX	0x00001800U

/* Maximum number of regions in one run. */
#define	LG_RUN_MAXREGS		11
#define	RUN_MAXREGS		(1U << LG_RUN_MAXREGS)

/*
 * Minimum redzone size.  Redzones may be larger than this if necessary to
 * preserve region alignment.
 */
#define	REDZONE_MINSIZE		16

/*
 * The minimum ratio of active:dirty pages per arena is computed as:
 *
 *   (nactive >> opt_lg_dirty_mult) >= ndirty
 *
 * So, supposing that opt_lg_dirty_mult is 3, there can be no less than 8 times
 * as many active pages as dirty pages.
 */
#define	LG_DIRTY_MULT_DEFAULT	3

typedef struct arena_chunk_map_s arena_chunk_map_t;
typedef struct arena_chunk_s arena_chunk_t;
typedef struct arena_run_s arena_run_t;
typedef struct arena_bin_info_s arena_bin_info_t;
typedef struct arena_bin_s arena_bin_t;
typedef struct arena_s arena_t;

#endif /* JEMALLOC_H_TYPES */
/******************************************************************************/
#ifdef JEMALLOC_H_STRUCTS

/* Each element of the chunk map corresponds to one page within the chunk. */
struct arena_chunk_map_s {
#ifndef JEMALLOC_PROF
	/*
	 * Overlay prof_ctx in order to allow it to be referenced by dead code.
	 * Such antics aren't warranted for per arena data structures, but
	 * chunk map overhead accounts for a percentage of memory, rather than
	 * being just a fixed cost.
	 */
	union {
#endif
	union {
		/*
		 * Linkage for run trees.  There are two disjoint uses:
		 *
		 * 1) arena_t's runs_avail tree.
		 * 2) arena_run_t conceptually uses this linkage for in-use
		 *    non-full runs, rather than directly embedding linkage.
		 */
		rb_node(arena_chunk_map_t)	rb_link;
		/*
		 * List of runs currently in purgatory.  arena_chunk_purge()
		 * temporarily allocates runs that contain dirty pages while
		 * purging, so that other threads cannot use the runs while the
		 * purging thread is operating without the arena lock held.
		 */
		ql_elm(arena_chunk_map_t)	ql_link;
	}				u;

	/* Profile counters, used for large object runs. */
	prof_ctx_t			*prof_ctx;
#ifndef JEMALLOC_PROF
	}; /* union { ... }; */
#endif

	/*
	 * Run address (or size) and various flags are stored together.  The bit
	 * layout looks like (assuming 32-bit system):
	 *
	 *   ???????? ???????? ????nnnn nnnndula
	 *
	 * ? : Unallocated: Run address for first/last pages, unset for internal
	 *                  pages.
	 *     Small: Run page offset.
	 *     Large: Run size for first page, unset for trailing pages.
	 * n : binind for small size class, BININD_INVALID for large size class.
	 * d : dirty?
	 * u : unzeroed?
	 * l : large?
	 * a : allocated?
	 *
	 * Following are example bit patterns for the three types of runs.
	 *
	 * p : run page offset
	 * s : run size
	 * n : binind for size class; large objects set these to BININD_INVALID
	 *     except for promoted allocations (see prof_promote)
	 * x : don't care
	 * - : 0
	 * + : 1
	 * [DULA] : bit set
	 * [dula] : bit unset
	 *
	 *   Unallocated (clean):
	 *     ssssssss ssssssss ssss++++ ++++du-a
	 *     xxxxxxxx xxxxxxxx xxxxxxxx xxxx-Uxx
	 *     ssssssss ssssssss ssss++++ ++++dU-a
	 *
	 *   Unallocated (dirty):
	 *     ssssssss ssssssss ssss++++ ++++D--a
	 *     xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx
	 *     ssssssss ssssssss ssss++++ ++++D--a
	 *
	 *   Small:
	 *     pppppppp pppppppp ppppnnnn nnnnd--A
	 *     pppppppp pppppppp ppppnnnn nnnn---A
	 *     pppppppp pppppppp ppppnnnn nnnnd--A
	 *
	 *   Large:
	 *     ssssssss ssssssss ssss++++ ++++D-LA
	 *     xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx
	 *     -------- -------- ----++++ ++++D-LA
	 *
	 *   Large (sampled, size <= PAGE):
	 *     ssssssss ssssssss ssssnnnn nnnnD-LA
	 *
	 *   Large (not sampled, size == PAGE):
	 *     ssssssss ssssssss ssss++++ ++++D-LA
	 */
	size_t				bits;
#define	CHUNK_MAP_BININD_SHIFT	4
#define	BININD_INVALID		((size_t)0xffU)
/*     CHUNK_MAP_BININD_MASK == (BININD_INVALID << CHUNK_MAP_BININD_SHIFT) */
#define	CHUNK_MAP_BININD_MASK	((size_t)0xff0U)
#define	CHUNK_MAP_BININD_INVALID CHUNK_MAP_BININD_MASK
#define	CHUNK_MAP_FLAGS_MASK	((size_t)0xcU)
#define	CHUNK_MAP_DIRTY		((size_t)0x8U)
#define	CHUNK_MAP_UNZEROED	((size_t)0x4U)
#define	CHUNK_MAP_LARGE		((size_t)0x2U)
#define	CHUNK_MAP_ALLOCATED	((size_t)0x1U)
#define	CHUNK_MAP_KEY		CHUNK_MAP_ALLOCATED
};
typedef rb_tree(arena_chunk_map_t) arena_avail_tree_t;
typedef rb_tree(arena_chunk_map_t) arena_run_tree_t;

/* Arena chunk header. */
struct arena_chunk_s {
	/* Arena that owns the chunk. */
	arena_t			*arena;

	/* Linkage for tree of arena chunks that contain dirty runs. */
	rb_node(arena_chunk_t)	dirty_link;

	/* Number of dirty pages. */
	size_t			ndirty;

	/* Number of available runs. */
	size_t			nruns_avail;

	/*
	 * Number of available run adjacencies.  Clean and dirty available runs
	 * are not coalesced, which causes virtual memory fragmentation.  The
	 * ratio of (nruns_avail-nruns_adjac):nruns_adjac is used for tracking
	 * this fragmentation.
	 * */
	size_t			nruns_adjac;

	/*
	 * Map of pages within chunk that keeps track of free/large/small.  The
	 * first map_bias entries are omitted, since the chunk header does not
	 * need to be tracked in the map.  This omission saves a header page
	 * for common chunk sizes (e.g. 4 MiB).
	 */
	arena_chunk_map_t	map[1]; /* Dynamically sized. */
};
typedef rb_tree(arena_chunk_t) arena_chunk_tree_t;

struct arena_run_s {
	/* Bin this run is associated with. */
	arena_bin_t	*bin;

	/* Index of next region that has never been allocated, or nregs. */
	uint32_t	nextind;

	/* Number of free regions in run. */
	unsigned	nfree;
};

/*
 * Read-only information associated with each element of arena_t's bins array
 * is stored separately, partly to reduce memory usage (only one copy, rather
 * than one per arena), but mainly to avoid false cacheline sharing.
 *
 * Each run has the following layout:
 *
 *               /--------------------\
 *               | arena_run_t header |
 *               | ...                |
 * bitmap_offset | bitmap             |
 *               | ...                |
 *   ctx0_offset | ctx map            |
 *               | ...                |
 *               |--------------------|
 *               | redzone            |
 *   reg0_offset | region 0           |
 *               | redzone            |
 *               |--------------------| \
 *               | redzone            | |
 *               | region 1           |  > reg_interval
 *               | redzone            | /
 *               |--------------------|
 *               | ...                |
 *               | ...                |
 *               | ...                |
 *               |--------------------|
 *               | redzone            |
 *               | region nregs-1     |
 *               | redzone            |
 *               |--------------------|
 *               | alignment pad?     |
 *               \--------------------/
 *
 * reg_interval has at least the same minimum alignment as reg_size; this
 * preserves the alignment constraint that sa2u() depends on.  Alignment pad is
 * either 0 or redzone_size; it is present only if needed to align reg0_offset.
 */
struct arena_bin_info_s {
	/* Size of regions in a run for this bin's size class. */
	size_t		reg_size;

	/* Redzone size. */
	size_t		redzone_size;

	/* Interval between regions (reg_size + (redzone_size << 1)). */
	size_t		reg_interval;

	/* Total size of a run for this bin's size class. */
	size_t		run_size;

	/* Total number of regions in a run for this bin's size class. */
	uint32_t	nregs;

	/*
	 * Offset of first bitmap_t element in a run header for this bin's size
	 * class.
	 */
	uint32_t	bitmap_offset;

	/*
	 * Metadata used to manipulate bitmaps for runs associated with this
	 * bin.
	 */
	bitmap_info_t	bitmap_info;

	/*
	 * Offset of first (prof_ctx_t *) in a run header for this bin's size
	 * class, or 0 if (config_prof == false || opt_prof == false).
	 */
	uint32_t	ctx0_offset;

	/* Offset of first region in a run for this bin's size class. */
	uint32_t	reg0_offset;
};

struct arena_bin_s {
	/*
	 * All operations on runcur, runs, and stats require that lock be
	 * locked.  Run allocation/deallocation are protected by the arena lock,
	 * which may be acquired while holding one or more bin locks, but not
	 * vise versa.
	 */
	malloc_mutex_t	lock;

	/*
	 * Current run being used to service allocations of this bin's size
	 * class.
	 */
	arena_run_t	*runcur;

	/*
	 * Tree of non-full runs.  This tree is used when looking for an
	 * existing run when runcur is no longer usable.  We choose the
	 * non-full run that is lowest in memory; this policy tends to keep
	 * objects packed well, and it can also help reduce the number of
	 * almost-empty chunks.
	 */
	arena_run_tree_t runs;

	/* Bin statistics. */
	malloc_bin_stats_t stats;
};

struct arena_s {
	/* This arena's index within the arenas array. */
	unsigned		ind;

	/*
	 * Number of threads currently assigned to this arena.  This field is
	 * protected by arenas_lock.
	 */
	unsigned		nthreads;

	/*
	 * There are three classes of arena operations from a locking
	 * perspective:
	 * 1) Thread asssignment (modifies nthreads) is protected by
	 *    arenas_lock.
	 * 2) Bin-related operations are protected by bin locks.
	 * 3) Chunk- and run-related operations are protected by this mutex.
	 */
	malloc_mutex_t		lock;

	arena_stats_t		stats;
	/*
	 * List of tcaches for extant threads associated with this arena.
	 * Stats from these are merged incrementally, and at exit.
	 */
	ql_head(tcache_t)	tcache_ql;

	uint64_t		prof_accumbytes;

	dss_prec_t		dss_prec;

	/* Tree of dirty-page-containing chunks this arena manages. */
	arena_chunk_tree_t	chunks_dirty;

	/*
	 * In order to avoid rapid chunk allocation/deallocation when an arena
	 * oscillates right on the cusp of needing a new chunk, cache the most
	 * recently freed chunk.  The spare is left in the arena's chunk trees
	 * until it is deleted.
	 *
	 * There is one spare chunk per arena, rather than one spare total, in
	 * order to avoid interactions between multiple threads that could make
	 * a single spare inadequate.
	 */
	arena_chunk_t		*spare;

	/* Number of pages in active runs. */
	size_t			nactive;

	/*
	 * Current count of pages within unused runs that are potentially
	 * dirty, and for which madvise(... MADV_DONTNEED) has not been called.
	 * By tracking this, we can institute a limit on how much dirty unused
	 * memory is mapped for each arena.
	 */
	size_t			ndirty;

	/*
	 * Approximate number of pages being purged.  It is possible for
	 * multiple threads to purge dirty pages concurrently, and they use
	 * npurgatory to indicate the total number of pages all threads are
	 * attempting to purge.
	 */
	size_t			npurgatory;

	/*
	 * Size/address-ordered trees of this arena's available runs.  The trees
	 * are used for first-best-fit run allocation.
	 */
	arena_avail_tree_t	runs_avail;

	/* bins is used to store trees of free regions. */
	arena_bin_t		bins[NBINS];
};

#endif /* JEMALLOC_H_STRUCTS */
/******************************************************************************/
#ifdef JEMALLOC_H_EXTERNS

extern ssize_t	opt_lg_dirty_mult;
/*
 * small_size2bin is a compact lookup table that rounds request sizes up to
 * size classes.  In order to reduce cache footprint, the table is compressed,
 * and all accesses are via the SMALL_SIZE2BIN macro.
 */
extern uint8_t const	small_size2bin[];
#define	SMALL_SIZE2BIN(s)	(small_size2bin[(s-1) >> LG_TINY_MIN])

extern arena_bin_info_t	arena_bin_info[NBINS];

/* Number of large size classes. */
#define			nlclasses (chunk_npages - map_bias)

void	arena_purge_all(arena_t *arena);
void	arena_tcache_fill_small(arena_t *arena, tcache_bin_t *tbin,
    size_t binind, uint64_t prof_accumbytes);
void	arena_alloc_junk_small(void *ptr, arena_bin_info_t *bin_info,
    bool zero);
void	arena_dalloc_junk_small(void *ptr, arena_bin_info_t *bin_info);
void	*arena_malloc_small(arena_t *arena, size_t size, bool zero);
void	*arena_malloc_large(arena_t *arena, size_t size, bool zero);
void	*arena_palloc(arena_t *arena, size_t size, size_t alignment, bool zero);
void	arena_prof_promoted(const void *ptr, size_t size);
void	arena_dalloc_bin_locked(arena_t *arena, arena_chunk_t *chunk, void *ptr,
    arena_chunk_map_t *mapelm);
void	arena_dalloc_bin(arena_t *arena, arena_chunk_t *chunk, void *ptr,
    size_t pageind, arena_chunk_map_t *mapelm);
void	arena_dalloc_small(arena_t *arena, arena_chunk_t *chunk, void *ptr,
    size_t pageind);
void	arena_dalloc_large_locked(arena_t *arena, arena_chunk_t *chunk,
    void *ptr);
void	arena_dalloc_large(arena_t *arena, arena_chunk_t *chunk, void *ptr);
void	*arena_ralloc_no_move(void *ptr, size_t oldsize, size_t size,
    size_t extra, bool zero);
void	*arena_ralloc(arena_t *arena, void *ptr, size_t oldsize, size_t size,
    size_t extra, size_t alignment, bool zero, bool try_tcache_alloc,
    bool try_tcache_dalloc);
dss_prec_t	arena_dss_prec_get(arena_t *arena);
void	arena_dss_prec_set(arena_t *arena, dss_prec_t dss_prec);
void	arena_stats_merge(arena_t *arena, const char **dss, size_t *nactive,
    size_t *ndirty, arena_stats_t *astats, malloc_bin_stats_t *bstats,
    malloc_large_stats_t *lstats);
bool	arena_new(arena_t *arena, unsigned ind);
void	arena_boot(void);
void	arena_prefork(arena_t *arena);
void	arena_postfork_parent(arena_t *arena);
void	arena_postfork_child(arena_t *arena);

#endif /* JEMALLOC_H_EXTERNS */
/******************************************************************************/
#ifdef JEMALLOC_H_INLINES

#ifndef JEMALLOC_ENABLE_INLINE
arena_chunk_map_t	*arena_mapp_get(arena_chunk_t *chunk, size_t pageind);
size_t	*arena_mapbitsp_get(arena_chunk_t *chunk, size_t pageind);
size_t	arena_mapbits_get(arena_chunk_t *chunk, size_t pageind);
size_t	arena_mapbits_unallocated_size_get(arena_chunk_t *chunk,
    size_t pageind);
size_t	arena_mapbits_large_size_get(arena_chunk_t *chunk, size_t pageind);
size_t	arena_mapbits_small_runind_get(arena_chunk_t *chunk, size_t pageind);
size_t	arena_mapbits_binind_get(arena_chunk_t *chunk, size_t pageind);
size_t	arena_mapbits_dirty_get(arena_chunk_t *chunk, size_t pageind);
size_t	arena_mapbits_unzeroed_get(arena_chunk_t *chunk, size_t pageind);
size_t	arena_mapbits_large_get(arena_chunk_t *chunk, size_t pageind);
size_t	arena_mapbits_allocated_get(arena_chunk_t *chunk, size_t pageind);
void	arena_mapbits_unallocated_set(arena_chunk_t *chunk, size_t pageind,
    size_t size, size_t flags);
void	arena_mapbits_unallocated_size_set(arena_chunk_t *chunk, size_t pageind,
    size_t size);
void	arena_mapbits_large_set(arena_chunk_t *chunk, size_t pageind,
    size_t size, size_t flags);
void	arena_mapbits_large_binind_set(arena_chunk_t *chunk, size_t pageind,
    size_t binind);
void	arena_mapbits_small_set(arena_chunk_t *chunk, size_t pageind,
    size_t runind, size_t binind, size_t flags);
void	arena_mapbits_unzeroed_set(arena_chunk_t *chunk, size_t pageind,
    size_t unzeroed);
bool	arena_prof_accum_impl(arena_t *arena, uint64_t accumbytes);
bool	arena_prof_accum_locked(arena_t *arena, uint64_t accumbytes);
bool	arena_prof_accum(arena_t *arena, uint64_t accumbytes);
size_t	arena_ptr_small_binind_get(const void *ptr, size_t mapbits);
size_t	arena_bin_index(arena_t *arena, arena_bin_t *bin);
unsigned	arena_run_regind(arena_run_t *run, arena_bin_info_t *bin_info,
    const void *ptr);
prof_ctx_t	*arena_prof_ctx_get(const void *ptr);
void	arena_prof_ctx_set(const void *ptr, prof_ctx_t *ctx);
void	*arena_malloc(arena_t *arena, size_t size, bool zero, bool try_tcache);
size_t	arena_salloc(const void *ptr, bool demote);
void	arena_dalloc(arena_t *arena, arena_chunk_t *chunk, void *ptr,
    bool try_tcache);
#endif

#if (defined(JEMALLOC_ENABLE_INLINE) || defined(JEMALLOC_ARENA_C_))
#  ifdef JEMALLOC_ARENA_INLINE_A
JEMALLOC_ALWAYS_INLINE arena_chunk_map_t *
arena_mapp_get(arena_chunk_t *chunk, size_t pageind)
{

	assert(pageind >= map_bias);
	assert(pageind < chunk_npages);

	return (&chunk->map[pageind-map_bias]);
}

JEMALLOC_ALWAYS_INLINE size_t *
arena_mapbitsp_get(arena_chunk_t *chunk, size_t pageind)
{

	return (&arena_mapp_get(chunk, pageind)->bits);
}

JEMALLOC_ALWAYS_INLINE size_t
arena_mapbits_get(arena_chunk_t *chunk, size_t pageind)
{

	return (*arena_mapbitsp_get(chunk, pageind));
}

JEMALLOC_ALWAYS_INLINE size_t
arena_mapbits_unallocated_size_get(arena_chunk_t *chunk, size_t pageind)
{
	size_t mapbits;

	mapbits = arena_mapbits_get(chunk, pageind);
	assert((mapbits & (CHUNK_MAP_LARGE|CHUNK_MAP_ALLOCATED)) == 0);
	return (mapbits & ~PAGE_MASK);
}

JEMALLOC_ALWAYS_INLINE size_t
arena_mapbits_large_size_get(arena_chunk_t *chunk, size_t pageind)
{
	size_t mapbits;

	mapbits = arena_mapbits_get(chunk, pageind);
	assert((mapbits & (CHUNK_MAP_LARGE|CHUNK_MAP_ALLOCATED)) ==
	    (CHUNK_MAP_LARGE|CHUNK_MAP_ALLOCATED));
	return (mapbits & ~PAGE_MASK);
}

JEMALLOC_ALWAYS_INLINE size_t
arena_mapbits_small_runind_get(arena_chunk_t *chunk, size_t pageind)
{
	size_t mapbits;

	mapbits = arena_mapbits_get(chunk, pageind);
	assert((mapbits & (CHUNK_MAP_LARGE|CHUNK_MAP_ALLOCATED)) ==
	    CHUNK_MAP_ALLOCATED);
	return (mapbits >> LG_PAGE);
}

JEMALLOC_ALWAYS_INLINE size_t
arena_mapbits_binind_get(arena_chunk_t *chunk, size_t pageind)
{
	size_t mapbits;
	size_t binind;

	mapbits = arena_mapbits_get(chunk, pageind);
	binind = (mapbits & CHUNK_MAP_BININD_MASK) >> CHUNK_MAP_BININD_SHIFT;
	assert(binind < NBINS || binind == BININD_INVALID);
	return (binind);
}

JEMALLOC_ALWAYS_INLINE size_t
arena_mapbits_dirty_get(arena_chunk_t *chunk, size_t pageind)
{
	size_t mapbits;

	mapbits = arena_mapbits_get(chunk, pageind);
	return (mapbits & CHUNK_MAP_DIRTY);
}

JEMALLOC_ALWAYS_INLINE size_t
arena_mapbits_unzeroed_get(arena_chunk_t *chunk, size_t pageind)
{
	size_t mapbits;

	mapbits = arena_mapbits_get(chunk, pageind);
	return (mapbits & CHUNK_MAP_UNZEROED);
}

JEMALLOC_ALWAYS_INLINE size_t
arena_mapbits_large_get(arena_chunk_t *chunk, size_t pageind)
{
	size_t mapbits;

	mapbits = arena_mapbits_get(chunk, pageind);
	return (mapbits & CHUNK_MAP_LARGE);
}

JEMALLOC_ALWAYS_INLINE size_t
arena_mapbits_allocated_get(arena_chunk_t *chunk, size_t pageind)
{
	size_t mapbits;

	mapbits = arena_mapbits_get(chunk, pageind);
	return (mapbits & CHUNK_MAP_ALLOCATED);
}

JEMALLOC_ALWAYS_INLINE void
arena_mapbits_unallocated_set(arena_chunk_t *chunk, size_t pageind, size_t size,
    size_t flags)
{
	size_t *mapbitsp;

	mapbitsp = arena_mapbitsp_get(chunk, pageind);
	assert((size & PAGE_MASK) == 0);
	assert((flags & ~CHUNK_MAP_FLAGS_MASK) == 0);
	assert((flags & (CHUNK_MAP_DIRTY|CHUNK_MAP_UNZEROED)) == flags);
	*mapbitsp = size | CHUNK_MAP_BININD_INVALID | flags;
}

JEMALLOC_ALWAYS_INLINE void
arena_mapbits_unallocated_size_set(arena_chunk_t *chunk, size_t pageind,
    size_t size)
{
	size_t *mapbitsp;

	mapbitsp = arena_mapbitsp_get(chunk, pageind);
	assert((size & PAGE_MASK) == 0);
	assert((*mapbitsp & (CHUNK_MAP_LARGE|CHUNK_MAP_ALLOCATED)) == 0);
	*mapbitsp = size | (*mapbitsp & PAGE_MASK);
}

JEMALLOC_ALWAYS_INLINE void
arena_mapbits_large_set(arena_chunk_t *chunk, size_t pageind, size_t size,
    size_t flags)
{
	size_t *mapbitsp;
	size_t unzeroed;

	mapbitsp = arena_mapbitsp_get(chunk, pageind);
	assert((size & PAGE_MASK) == 0);
	assert((flags & CHUNK_MAP_DIRTY) == flags);
	unzeroed = *mapbitsp & CHUNK_MAP_UNZEROED; /* Preserve unzeroed. */
	*mapbitsp = size | CHUNK_MAP_BININD_INVALID | flags | unzeroed |
	    CHUNK_MAP_LARGE | CHUNK_MAP_ALLOCATED;
}

JEMALLOC_ALWAYS_INLINE void
arena_mapbits_large_binind_set(arena_chunk_t *chunk, size_t pageind,
    size_t binind)
{
	size_t *mapbitsp;

	assert(binind <= BININD_INVALID);
	mapbitsp = arena_mapbitsp_get(chunk, pageind);
	assert(arena_mapbits_large_size_get(chunk, pageind) == PAGE);
	*mapbitsp = (*mapbitsp & ~CHUNK_MAP_BININD_MASK) | (binind <<
	    CHUNK_MAP_BININD_SHIFT);
}

JEMALLOC_ALWAYS_INLINE void
arena_mapbits_small_set(arena_chunk_t *chunk, size_t pageind, size_t runind,
    size_t binind, size_t flags)
{
	size_t *mapbitsp;
	size_t unzeroed;

	assert(binind < BININD_INVALID);
	mapbitsp = arena_mapbitsp_get(chunk, pageind);
	assert(pageind - runind >= map_bias);
	assert((flags & CHUNK_MAP_DIRTY) == flags);
	unzeroed = *mapbitsp & CHUNK_MAP_UNZEROED; /* Preserve unzeroed. */
	*mapbitsp = (runind << LG_PAGE) | (binind << CHUNK_MAP_BININD_SHIFT) |
	    flags | unzeroed | CHUNK_MAP_ALLOCATED;
}

JEMALLOC_ALWAYS_INLINE void
arena_mapbits_unzeroed_set(arena_chunk_t *chunk, size_t pageind,
    size_t unzeroed)
{
	size_t *mapbitsp;

	mapbitsp = arena_mapbitsp_get(chunk, pageind);
	*mapbitsp = (*mapbitsp & ~CHUNK_MAP_UNZEROED) | unzeroed;
}

JEMALLOC_INLINE bool
arena_prof_accum_impl(arena_t *arena, uint64_t accumbytes)
{

	cassert(config_prof);
	assert(prof_interval != 0);

	arena->prof_accumbytes += accumbytes;
	if (arena->prof_accumbytes >= prof_interval) {
		arena->prof_accumbytes -= prof_interval;
		return (true);
	}
	return (false);
}

JEMALLOC_INLINE bool
arena_prof_accum_locked(arena_t *arena, uint64_t accumbytes)
{

	cassert(config_prof);

	if (prof_interval == 0)
		return (false);
	return (arena_prof_accum_impl(arena, accumbytes));
}

JEMALLOC_INLINE bool
arena_prof_accum(arena_t *arena, uint64_t accumbytes)
{

	cassert(config_prof);

	if (prof_interval == 0)
		return (false);

	{
		bool ret;

		malloc_mutex_lock(&arena->lock);
		ret = arena_prof_accum_impl(arena, accumbytes);
		malloc_mutex_unlock(&arena->lock);
		return (ret);
	}
}

JEMALLOC_ALWAYS_INLINE size_t
arena_ptr_small_binind_get(const void *ptr, size_t mapbits)
{
	size_t binind;

	binind = (mapbits & CHUNK_MAP_BININD_MASK) >> CHUNK_MAP_BININD_SHIFT;

	if (config_debug) {
		arena_chunk_t *chunk;
		arena_t *arena;
		size_t pageind;
		size_t actual_mapbits;
		arena_run_t *run;
		arena_bin_t *bin;
		size_t actual_binind;
		arena_bin_info_t *bin_info;

		assert(binind != BININD_INVALID);
		assert(binind < NBINS);
		chunk = (arena_chunk_t *)CHUNK_ADDR2BASE(ptr);
		arena = chunk->arena;
		pageind = ((uintptr_t)ptr - (uintptr_t)chunk) >> LG_PAGE;
		actual_mapbits = arena_mapbits_get(chunk, pageind);
		assert(mapbits == actual_mapbits);
		assert(arena_mapbits_large_get(chunk, pageind) == 0);
		assert(arena_mapbits_allocated_get(chunk, pageind) != 0);
		run = (arena_run_t *)((uintptr_t)chunk + (uintptr_t)((pageind -
		    (actual_mapbits >> LG_PAGE)) << LG_PAGE));
		bin = run->bin;
		actual_binind = bin - arena->bins;
		assert(binind == actual_binind);
		bin_info = &arena_bin_info[actual_binind];
		assert(((uintptr_t)ptr - ((uintptr_t)run +
		    (uintptr_t)bin_info->reg0_offset)) % bin_info->reg_interval
		    == 0);
	}

	return (binind);
}
#  endif /* JEMALLOC_ARENA_INLINE_A */

#  ifdef JEMALLOC_ARENA_INLINE_B
JEMALLOC_INLINE size_t
arena_bin_index(arena_t *arena, arena_bin_t *bin)
{
	size_t binind = bin - arena->bins;
	assert(binind < NBINS);
	return (binind);
}

JEMALLOC_INLINE unsigned
arena_run_regind(arena_run_t *run, arena_bin_info_t *bin_info, const void *ptr)
{
	unsigned shift, diff, regind;
	size_t interval;

	/*
	 * Freeing a pointer lower than region zero can cause assertion
	 * failure.
	 */
	assert((uintptr_t)ptr >= (uintptr_t)run +
	    (uintptr_t)bin_info->reg0_offset);

	/*
	 * Avoid doing division with a variable divisor if possible.  Using
	 * actual division here can reduce allocator throughput by over 20%!
	 */
	diff = (unsigned)((uintptr_t)ptr - (uintptr_t)run -
	    bin_info->reg0_offset);

	/* Rescale (factor powers of 2 out of the numerator and denominator). */
	interval = bin_info->reg_interval;
	shift = ffs(interval) - 1;
	diff >>= shift;
	interval >>= shift;

	if (interval == 1) {
		/* The divisor was a power of 2. */
		regind = diff;
	} else {
		/*
		 * To divide by a number D that is not a power of two we
		 * multiply by (2^21 / D) and then right shift by 21 positions.
		 *
		 *   X / D
		 *
		 * becomes
		 *
		 *   (X * interval_invs[D - 3]) >> SIZE_INV_SHIFT
		 *
		 * We can omit the first three elements, because we never
		 * divide by 0, and 1 and 2 are both powers of two, which are
		 * handled above.
		 */
#define	SIZE_INV_SHIFT	((sizeof(unsigned) << 3) - LG_RUN_MAXREGS)
#define	SIZE_INV(s)	(((1U << SIZE_INV_SHIFT) / (s)) + 1)
		static const unsigned interval_invs[] = {
		    SIZE_INV(3),
		    SIZE_INV(4), SIZE_INV(5), SIZE_INV(6), SIZE_INV(7),
		    SIZE_INV(8), SIZE_INV(9), SIZE_INV(10), SIZE_INV(11),
		    SIZE_INV(12), SIZE_INV(13), SIZE_INV(14), SIZE_INV(15),
		    SIZE_INV(16), SIZE_INV(17), SIZE_INV(18), SIZE_INV(19),
		    SIZE_INV(20), SIZE_INV(21), SIZE_INV(22), SIZE_INV(23),
		    SIZE_INV(24), SIZE_INV(25), SIZE_INV(26), SIZE_INV(27),
		    SIZE_INV(28), SIZE_INV(29), SIZE_INV(30), SIZE_INV(31)
		};

		if (interval <= ((sizeof(interval_invs) / sizeof(unsigned)) +
		    2)) {
			regind = (diff * interval_invs[interval - 3]) >>
			    SIZE_INV_SHIFT;
		} else
			regind = diff / interval;
#undef SIZE_INV
#undef SIZE_INV_SHIFT
	}
	assert(diff == regind * interval);
	assert(regind < bin_info->nregs);

	return (regind);
}

JEMALLOC_INLINE prof_ctx_t *
arena_prof_ctx_get(const void *ptr)
{
	prof_ctx_t *ret;
	arena_chunk_t *chunk;
	size_t pageind, mapbits;

	cassert(config_prof);
	assert(ptr != NULL);
	assert(CHUNK_ADDR2BASE(ptr) != ptr);

	chunk = (arena_chunk_t *)CHUNK_ADDR2BASE(ptr);
	pageind = ((uintptr_t)ptr - (uintptr_t)chunk) >> LG_PAGE;
	mapbits = arena_mapbits_get(chunk, pageind);
	assert((mapbits & CHUNK_MAP_ALLOCATED) != 0);
	if ((mapbits & CHUNK_MAP_LARGE) == 0) {
		if (prof_promote)
			ret = (prof_ctx_t *)(uintptr_t)1U;
		else {
			arena_run_t *run = (arena_run_t *)((uintptr_t)chunk +
			    (uintptr_t)((pageind - (mapbits >> LG_PAGE)) <<
			    LG_PAGE));
			size_t binind = arena_ptr_small_binind_get(ptr,
			    mapbits);
			arena_bin_info_t *bin_info = &arena_bin_info[binind];
			unsigned regind;

			regind = arena_run_regind(run, bin_info, ptr);
			ret = *(prof_ctx_t **)((uintptr_t)run +
			    bin_info->ctx0_offset + (regind *
			    sizeof(prof_ctx_t *)));
		}
	} else
		ret = arena_mapp_get(chunk, pageind)->prof_ctx;

	return (ret);
}

JEMALLOC_INLINE void
arena_prof_ctx_set(const void *ptr, prof_ctx_t *ctx)
{
	arena_chunk_t *chunk;
	size_t pageind, mapbits;

	cassert(config_prof);
	assert(ptr != NULL);
	assert(CHUNK_ADDR2BASE(ptr) != ptr);

	chunk = (arena_chunk_t *)CHUNK_ADDR2BASE(ptr);
	pageind = ((uintptr_t)ptr - (uintptr_t)chunk) >> LG_PAGE;
	mapbits = arena_mapbits_get(chunk, pageind);
	assert((mapbits & CHUNK_MAP_ALLOCATED) != 0);
	if ((mapbits & CHUNK_MAP_LARGE) == 0) {
		if (prof_promote == false) {
			arena_run_t *run = (arena_run_t *)((uintptr_t)chunk +
			    (uintptr_t)((pageind - (mapbits >> LG_PAGE)) <<
			    LG_PAGE));
			size_t binind;
			arena_bin_info_t *bin_info;
			unsigned regind;

			binind = arena_ptr_small_binind_get(ptr, mapbits);
			bin_info = &arena_bin_info[binind];
			regind = arena_run_regind(run, bin_info, ptr);

			*((prof_ctx_t **)((uintptr_t)run + bin_info->ctx0_offset
			    + (regind * sizeof(prof_ctx_t *)))) = ctx;
		} else
			assert((uintptr_t)ctx == (uintptr_t)1U);
	} else
		arena_mapp_get(chunk, pageind)->prof_ctx = ctx;
}

JEMALLOC_ALWAYS_INLINE void *
arena_malloc(arena_t *arena, size_t size, bool zero, bool try_tcache)
{
	tcache_t *tcache;

	assert(size != 0);
	assert(size <= arena_maxclass);

	if (size <= SMALL_MAXCLASS) {
		if (try_tcache && (tcache = tcache_get(true)) != NULL)
			return (tcache_alloc_small(tcache, size, zero));
		else {
			return (arena_malloc_small(choose_arena(arena), size,
			    zero));
		}
	} else {
		/*
		 * Initialize tcache after checking size in order to avoid
		 * infinite recursion during tcache initialization.
		 */
		if (try_tcache && size <= tcache_maxclass && (tcache =
		    tcache_get(true)) != NULL)
			return (tcache_alloc_large(tcache, size, zero));
		else {
			return (arena_malloc_large(choose_arena(arena), size,
			    zero));
		}
	}
}

/* Return the size of the allocation pointed to by ptr. */
JEMALLOC_ALWAYS_INLINE size_t
arena_salloc(const void *ptr, bool demote)
{
	size_t ret;
	arena_chunk_t *chunk;
	size_t pageind, binind;

	assert(ptr != NULL);
	assert(CHUNK_ADDR2BASE(ptr) != ptr);

	chunk = (arena_chunk_t *)CHUNK_ADDR2BASE(ptr);
	pageind = ((uintptr_t)ptr - (uintptr_t)chunk) >> LG_PAGE;
	assert(arena_mapbits_allocated_get(chunk, pageind) != 0);
	binind = arena_mapbits_binind_get(chunk, pageind);
	if (binind == BININD_INVALID || (config_prof && demote == false &&
	    prof_promote && arena_mapbits_large_get(chunk, pageind) != 0)) {
		/*
		 * Large allocation.  In the common case (demote == true), and
		 * as this is an inline function, most callers will only end up
		 * looking at binind to determine that ptr is a small
		 * allocation.
		 */
		assert(((uintptr_t)ptr & PAGE_MASK) == 0);
		ret = arena_mapbits_large_size_get(chunk, pageind);
		assert(ret != 0);
		assert(pageind + (ret>>LG_PAGE) <= chunk_npages);
		assert(ret == PAGE || arena_mapbits_large_size_get(chunk,
		    pageind+(ret>>LG_PAGE)-1) == 0);
		assert(binind == arena_mapbits_binind_get(chunk,
		    pageind+(ret>>LG_PAGE)-1));
		assert(arena_mapbits_dirty_get(chunk, pageind) ==
		    arena_mapbits_dirty_get(chunk, pageind+(ret>>LG_PAGE)-1));
	} else {
		/*
		 * Small allocation (possibly promoted to a large object due to
		 * prof_promote).
		 */
		assert(arena_mapbits_large_get(chunk, pageind) != 0 ||
		    arena_ptr_small_binind_get(ptr, arena_mapbits_get(chunk,
		    pageind)) == binind);
		ret = arena_bin_info[binind].reg_size;
	}

	return (ret);
}

JEMALLOC_ALWAYS_INLINE void
arena_dalloc(arena_t *arena, arena_chunk_t *chunk, void *ptr, bool try_tcache)
{
	size_t pageind, mapbits;
	tcache_t *tcache;

	assert(arena != NULL);
	assert(chunk->arena == arena);
	assert(ptr != NULL);
	assert(CHUNK_ADDR2BASE(ptr) != ptr);

	pageind = ((uintptr_t)ptr - (uintptr_t)chunk) >> LG_PAGE;
	mapbits = arena_mapbits_get(chunk, pageind);
	assert(arena_mapbits_allocated_get(chunk, pageind) != 0);
	if ((mapbits & CHUNK_MAP_LARGE) == 0) {
		/* Small allocation. */
		if (try_tcache && (tcache = tcache_get(false)) != NULL) {
			size_t binind;

			binind = arena_ptr_small_binind_get(ptr, mapbits);
			tcache_dalloc_small(tcache, ptr, binind);
		} else
			arena_dalloc_small(arena, chunk, ptr, pageind);
	} else {
		size_t size = arena_mapbits_large_size_get(chunk, pageind);

		assert(((uintptr_t)ptr & PAGE_MASK) == 0);

		if (try_tcache && size <= tcache_maxclass && (tcache =
		    tcache_get(false)) != NULL) {
			tcache_dalloc_large(tcache, ptr, size);
		} else
			arena_dalloc_large(arena, chunk, ptr);
	}
}
#  endif /* JEMALLOC_ARENA_INLINE_B */
#endif

#endif /* JEMALLOC_H_INLINES */
/******************************************************************************/
