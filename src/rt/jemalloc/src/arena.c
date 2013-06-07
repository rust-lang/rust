#define	JEMALLOC_ARENA_C_
#include "jemalloc/internal/jemalloc_internal.h"

/******************************************************************************/
/* Data. */

ssize_t		opt_lg_dirty_mult = LG_DIRTY_MULT_DEFAULT;
arena_bin_info_t	arena_bin_info[NBINS];

JEMALLOC_ALIGNED(CACHELINE)
const uint8_t	small_size2bin[] = {
#define	S2B_8(i)	i,
#define	S2B_16(i)	S2B_8(i) S2B_8(i)
#define	S2B_32(i)	S2B_16(i) S2B_16(i)
#define	S2B_64(i)	S2B_32(i) S2B_32(i)
#define	S2B_128(i)	S2B_64(i) S2B_64(i)
#define	S2B_256(i)	S2B_128(i) S2B_128(i)
#define	S2B_512(i)	S2B_256(i) S2B_256(i)
#define	S2B_1024(i)	S2B_512(i) S2B_512(i)
#define	S2B_2048(i)	S2B_1024(i) S2B_1024(i)
#define	S2B_4096(i)	S2B_2048(i) S2B_2048(i)
#define	S2B_8192(i)	S2B_4096(i) S2B_4096(i)
#define	SIZE_CLASS(bin, delta, size)					\
	S2B_##delta(bin)
	SIZE_CLASSES
#undef S2B_8
#undef S2B_16
#undef S2B_32
#undef S2B_64
#undef S2B_128
#undef S2B_256
#undef S2B_512
#undef S2B_1024
#undef S2B_2048
#undef S2B_4096
#undef S2B_8192
#undef SIZE_CLASS
};

/******************************************************************************/
/* Function prototypes for non-inline static functions. */

static void	arena_avail_insert(arena_t *arena, arena_chunk_t *chunk,
    size_t pageind, size_t npages, bool maybe_adjac_pred,
    bool maybe_adjac_succ);
static void	arena_avail_remove(arena_t *arena, arena_chunk_t *chunk,
    size_t pageind, size_t npages, bool maybe_adjac_pred,
    bool maybe_adjac_succ);
static void	arena_run_split(arena_t *arena, arena_run_t *run, size_t size,
    bool large, size_t binind, bool zero);
static arena_chunk_t *arena_chunk_alloc(arena_t *arena);
static void	arena_chunk_dealloc(arena_t *arena, arena_chunk_t *chunk);
static arena_run_t	*arena_run_alloc_helper(arena_t *arena, size_t size,
    bool large, size_t binind, bool zero);
static arena_run_t *arena_run_alloc(arena_t *arena, size_t size, bool large,
    size_t binind, bool zero);
static arena_chunk_t	*chunks_dirty_iter_cb(arena_chunk_tree_t *tree,
    arena_chunk_t *chunk, void *arg);
static void	arena_purge(arena_t *arena, bool all);
static void	arena_run_dalloc(arena_t *arena, arena_run_t *run, bool dirty,
    bool cleaned);
static void	arena_run_trim_head(arena_t *arena, arena_chunk_t *chunk,
    arena_run_t *run, size_t oldsize, size_t newsize);
static void	arena_run_trim_tail(arena_t *arena, arena_chunk_t *chunk,
    arena_run_t *run, size_t oldsize, size_t newsize, bool dirty);
static arena_run_t	*arena_bin_runs_first(arena_bin_t *bin);
static void	arena_bin_runs_insert(arena_bin_t *bin, arena_run_t *run);
static void	arena_bin_runs_remove(arena_bin_t *bin, arena_run_t *run);
static arena_run_t *arena_bin_nonfull_run_tryget(arena_bin_t *bin);
static arena_run_t *arena_bin_nonfull_run_get(arena_t *arena, arena_bin_t *bin);
static void	*arena_bin_malloc_hard(arena_t *arena, arena_bin_t *bin);
static void	arena_dissociate_bin_run(arena_chunk_t *chunk, arena_run_t *run,
    arena_bin_t *bin);
static void	arena_dalloc_bin_run(arena_t *arena, arena_chunk_t *chunk,
    arena_run_t *run, arena_bin_t *bin);
static void	arena_bin_lower_run(arena_t *arena, arena_chunk_t *chunk,
    arena_run_t *run, arena_bin_t *bin);
static void	arena_ralloc_large_shrink(arena_t *arena, arena_chunk_t *chunk,
    void *ptr, size_t oldsize, size_t size);
static bool	arena_ralloc_large_grow(arena_t *arena, arena_chunk_t *chunk,
    void *ptr, size_t oldsize, size_t size, size_t extra, bool zero);
static bool	arena_ralloc_large(void *ptr, size_t oldsize, size_t size,
    size_t extra, bool zero);
static size_t	bin_info_run_size_calc(arena_bin_info_t *bin_info,
    size_t min_run_size);
static void	bin_info_init(void);

/******************************************************************************/

static inline int
arena_run_comp(arena_chunk_map_t *a, arena_chunk_map_t *b)
{
	uintptr_t a_mapelm = (uintptr_t)a;
	uintptr_t b_mapelm = (uintptr_t)b;

	assert(a != NULL);
	assert(b != NULL);

	return ((a_mapelm > b_mapelm) - (a_mapelm < b_mapelm));
}

/* Generate red-black tree functions. */
rb_gen(static UNUSED, arena_run_tree_, arena_run_tree_t, arena_chunk_map_t,
    u.rb_link, arena_run_comp)

static inline int
arena_avail_comp(arena_chunk_map_t *a, arena_chunk_map_t *b)
{
	int ret;
	size_t a_size = a->bits & ~PAGE_MASK;
	size_t b_size = b->bits & ~PAGE_MASK;

	ret = (a_size > b_size) - (a_size < b_size);
	if (ret == 0) {
		uintptr_t a_mapelm, b_mapelm;

		if ((a->bits & CHUNK_MAP_KEY) != CHUNK_MAP_KEY)
			a_mapelm = (uintptr_t)a;
		else {
			/*
			 * Treat keys as though they are lower than anything
			 * else.
			 */
			a_mapelm = 0;
		}
		b_mapelm = (uintptr_t)b;

		ret = (a_mapelm > b_mapelm) - (a_mapelm < b_mapelm);
	}

	return (ret);
}

/* Generate red-black tree functions. */
rb_gen(static UNUSED, arena_avail_tree_, arena_avail_tree_t, arena_chunk_map_t,
    u.rb_link, arena_avail_comp)

static inline int
arena_chunk_dirty_comp(arena_chunk_t *a, arena_chunk_t *b)
{

	assert(a != NULL);
	assert(b != NULL);

	/*
	 * Short-circuit for self comparison.  The following comparison code
	 * would come to the same result, but at the cost of executing the slow
	 * path.
	 */
	if (a == b)
		return (0);

	/*
	 * Order such that chunks with higher fragmentation are "less than"
	 * those with lower fragmentation -- purging order is from "least" to
	 * "greatest".  Fragmentation is measured as:
	 *
	 *     mean current avail run size
	 *   --------------------------------
	 *   mean defragmented avail run size
	 *
	 *            navail
	 *         -----------
	 *         nruns_avail           nruns_avail-nruns_adjac
	 * = ========================= = -----------------------
	 *            navail                  nruns_avail
	 *    -----------------------
	 *    nruns_avail-nruns_adjac
	 *
	 * The following code multiplies away the denominator prior to
	 * comparison, in order to avoid division.
	 *
	 */
	{
		size_t a_val = (a->nruns_avail - a->nruns_adjac) *
		    b->nruns_avail;
		size_t b_val = (b->nruns_avail - b->nruns_adjac) *
		    a->nruns_avail;

		if (a_val < b_val)
			return (1);
		if (a_val > b_val)
			return (-1);
	}
	/*
	 * Break ties by chunk address.  For fragmented chunks, report lower
	 * addresses as "lower", so that fragmentation reduction happens first
	 * at lower addresses.  However, use the opposite ordering for
	 * unfragmented chunks, in order to increase the chances of
	 * re-allocating dirty runs.
	 */
	{
		uintptr_t a_chunk = (uintptr_t)a;
		uintptr_t b_chunk = (uintptr_t)b;
		int ret = ((a_chunk > b_chunk) - (a_chunk < b_chunk));
		if (a->nruns_adjac == 0) {
			assert(b->nruns_adjac == 0);
			ret = -ret;
		}
		return (ret);
	}
}

/* Generate red-black tree functions. */
rb_gen(static UNUSED, arena_chunk_dirty_, arena_chunk_tree_t, arena_chunk_t,
    dirty_link, arena_chunk_dirty_comp)

static inline bool
arena_avail_adjac_pred(arena_chunk_t *chunk, size_t pageind)
{
	bool ret;

	if (pageind-1 < map_bias)
		ret = false;
	else {
		ret = (arena_mapbits_allocated_get(chunk, pageind-1) == 0);
		assert(ret == false || arena_mapbits_dirty_get(chunk,
		    pageind-1) != arena_mapbits_dirty_get(chunk, pageind));
	}
	return (ret);
}

static inline bool
arena_avail_adjac_succ(arena_chunk_t *chunk, size_t pageind, size_t npages)
{
	bool ret;

	if (pageind+npages == chunk_npages)
		ret = false;
	else {
		assert(pageind+npages < chunk_npages);
		ret = (arena_mapbits_allocated_get(chunk, pageind+npages) == 0);
		assert(ret == false || arena_mapbits_dirty_get(chunk, pageind)
		    != arena_mapbits_dirty_get(chunk, pageind+npages));
	}
	return (ret);
}

static inline bool
arena_avail_adjac(arena_chunk_t *chunk, size_t pageind, size_t npages)
{

	return (arena_avail_adjac_pred(chunk, pageind) ||
	    arena_avail_adjac_succ(chunk, pageind, npages));
}

static void
arena_avail_insert(arena_t *arena, arena_chunk_t *chunk, size_t pageind,
    size_t npages, bool maybe_adjac_pred, bool maybe_adjac_succ)
{

	assert(npages == (arena_mapbits_unallocated_size_get(chunk, pageind) >>
	    LG_PAGE));

	/*
	 * chunks_dirty is keyed by nruns_{avail,adjac}, so the chunk must be
	 * removed and reinserted even if the run to be inserted is clean.
	 */
	if (chunk->ndirty != 0)
		arena_chunk_dirty_remove(&arena->chunks_dirty, chunk);

	if (maybe_adjac_pred && arena_avail_adjac_pred(chunk, pageind))
		chunk->nruns_adjac++;
	if (maybe_adjac_succ && arena_avail_adjac_succ(chunk, pageind, npages))
		chunk->nruns_adjac++;
	chunk->nruns_avail++;
	assert(chunk->nruns_avail > chunk->nruns_adjac);

	if (arena_mapbits_dirty_get(chunk, pageind) != 0) {
		arena->ndirty += npages;
		chunk->ndirty += npages;
	}
	if (chunk->ndirty != 0)
		arena_chunk_dirty_insert(&arena->chunks_dirty, chunk);

	arena_avail_tree_insert(&arena->runs_avail, arena_mapp_get(chunk,
	    pageind));
}

static void
arena_avail_remove(arena_t *arena, arena_chunk_t *chunk, size_t pageind,
    size_t npages, bool maybe_adjac_pred, bool maybe_adjac_succ)
{

	assert(npages == (arena_mapbits_unallocated_size_get(chunk, pageind) >>
	    LG_PAGE));

	/*
	 * chunks_dirty is keyed by nruns_{avail,adjac}, so the chunk must be
	 * removed and reinserted even if the run to be removed is clean.
	 */
	if (chunk->ndirty != 0)
		arena_chunk_dirty_remove(&arena->chunks_dirty, chunk);

	if (maybe_adjac_pred && arena_avail_adjac_pred(chunk, pageind))
		chunk->nruns_adjac--;
	if (maybe_adjac_succ && arena_avail_adjac_succ(chunk, pageind, npages))
		chunk->nruns_adjac--;
	chunk->nruns_avail--;
	assert(chunk->nruns_avail > chunk->nruns_adjac || (chunk->nruns_avail
	    == 0 && chunk->nruns_adjac == 0));

	if (arena_mapbits_dirty_get(chunk, pageind) != 0) {
		arena->ndirty -= npages;
		chunk->ndirty -= npages;
	}
	if (chunk->ndirty != 0)
		arena_chunk_dirty_insert(&arena->chunks_dirty, chunk);

	arena_avail_tree_remove(&arena->runs_avail, arena_mapp_get(chunk,
	    pageind));
}

static inline void *
arena_run_reg_alloc(arena_run_t *run, arena_bin_info_t *bin_info)
{
	void *ret;
	unsigned regind;
	bitmap_t *bitmap = (bitmap_t *)((uintptr_t)run +
	    (uintptr_t)bin_info->bitmap_offset);

	assert(run->nfree > 0);
	assert(bitmap_full(bitmap, &bin_info->bitmap_info) == false);

	regind = bitmap_sfu(bitmap, &bin_info->bitmap_info);
	ret = (void *)((uintptr_t)run + (uintptr_t)bin_info->reg0_offset +
	    (uintptr_t)(bin_info->reg_interval * regind));
	run->nfree--;
	if (regind == run->nextind)
		run->nextind++;
	assert(regind < run->nextind);
	return (ret);
}

static inline void
arena_run_reg_dalloc(arena_run_t *run, void *ptr)
{
	arena_chunk_t *chunk = (arena_chunk_t *)CHUNK_ADDR2BASE(run);
	size_t pageind = ((uintptr_t)ptr - (uintptr_t)chunk) >> LG_PAGE;
	size_t mapbits = arena_mapbits_get(chunk, pageind);
	size_t binind = arena_ptr_small_binind_get(ptr, mapbits);
	arena_bin_info_t *bin_info = &arena_bin_info[binind];
	unsigned regind = arena_run_regind(run, bin_info, ptr);
	bitmap_t *bitmap = (bitmap_t *)((uintptr_t)run +
	    (uintptr_t)bin_info->bitmap_offset);

	assert(run->nfree < bin_info->nregs);
	/* Freeing an interior pointer can cause assertion failure. */
	assert(((uintptr_t)ptr - ((uintptr_t)run +
	    (uintptr_t)bin_info->reg0_offset)) %
	    (uintptr_t)bin_info->reg_interval == 0);
	assert((uintptr_t)ptr >= (uintptr_t)run +
	    (uintptr_t)bin_info->reg0_offset);
	/* Freeing an unallocated pointer can cause assertion failure. */
	assert(bitmap_get(bitmap, &bin_info->bitmap_info, regind));

	bitmap_unset(bitmap, &bin_info->bitmap_info, regind);
	run->nfree++;
}

static inline void
arena_run_zero(arena_chunk_t *chunk, size_t run_ind, size_t npages)
{

	VALGRIND_MAKE_MEM_UNDEFINED((void *)((uintptr_t)chunk + (run_ind <<
	    LG_PAGE)), (npages << LG_PAGE));
	memset((void *)((uintptr_t)chunk + (run_ind << LG_PAGE)), 0,
	    (npages << LG_PAGE));
}

static inline void
arena_run_page_validate_zeroed(arena_chunk_t *chunk, size_t run_ind)
{
	size_t i;
	UNUSED size_t *p = (size_t *)((uintptr_t)chunk + (run_ind << LG_PAGE));

	VALGRIND_MAKE_MEM_DEFINED((void *)((uintptr_t)chunk + (run_ind <<
	    LG_PAGE)), PAGE);
	for (i = 0; i < PAGE / sizeof(size_t); i++)
		assert(p[i] == 0);
}

static void
arena_run_split(arena_t *arena, arena_run_t *run, size_t size, bool large,
    size_t binind, bool zero)
{
	arena_chunk_t *chunk;
	size_t run_ind, total_pages, need_pages, rem_pages, i;
	size_t flag_dirty;

	assert((large && binind == BININD_INVALID) || (large == false && binind
	    != BININD_INVALID));

	chunk = (arena_chunk_t *)CHUNK_ADDR2BASE(run);
	run_ind = (unsigned)(((uintptr_t)run - (uintptr_t)chunk) >> LG_PAGE);
	flag_dirty = arena_mapbits_dirty_get(chunk, run_ind);
	total_pages = arena_mapbits_unallocated_size_get(chunk, run_ind) >>
	    LG_PAGE;
	assert(arena_mapbits_dirty_get(chunk, run_ind+total_pages-1) ==
	    flag_dirty);
	need_pages = (size >> LG_PAGE);
	assert(need_pages > 0);
	assert(need_pages <= total_pages);
	rem_pages = total_pages - need_pages;

	arena_avail_remove(arena, chunk, run_ind, total_pages, true, true);
	if (config_stats) {
		/*
		 * Update stats_cactive if nactive is crossing a chunk
		 * multiple.
		 */
		size_t cactive_diff = CHUNK_CEILING((arena->nactive +
		    need_pages) << LG_PAGE) - CHUNK_CEILING(arena->nactive <<
		    LG_PAGE);
		if (cactive_diff != 0)
			stats_cactive_add(cactive_diff);
	}
	arena->nactive += need_pages;

	/* Keep track of trailing unused pages for later use. */
	if (rem_pages > 0) {
		if (flag_dirty != 0) {
			arena_mapbits_unallocated_set(chunk, run_ind+need_pages,
			    (rem_pages << LG_PAGE), CHUNK_MAP_DIRTY);
			arena_mapbits_unallocated_set(chunk,
			    run_ind+total_pages-1, (rem_pages << LG_PAGE),
			    CHUNK_MAP_DIRTY);
		} else {
			arena_mapbits_unallocated_set(chunk, run_ind+need_pages,
			    (rem_pages << LG_PAGE),
			    arena_mapbits_unzeroed_get(chunk,
			    run_ind+need_pages));
			arena_mapbits_unallocated_set(chunk,
			    run_ind+total_pages-1, (rem_pages << LG_PAGE),
			    arena_mapbits_unzeroed_get(chunk,
			    run_ind+total_pages-1));
		}
		arena_avail_insert(arena, chunk, run_ind+need_pages, rem_pages,
		    false, true);
	}

	/*
	 * Update the page map separately for large vs. small runs, since it is
	 * possible to avoid iteration for large mallocs.
	 */
	if (large) {
		if (zero) {
			if (flag_dirty == 0) {
				/*
				 * The run is clean, so some pages may be
				 * zeroed (i.e. never before touched).
				 */
				for (i = 0; i < need_pages; i++) {
					if (arena_mapbits_unzeroed_get(chunk,
					    run_ind+i) != 0) {
						arena_run_zero(chunk, run_ind+i,
						    1);
					} else if (config_debug) {
						arena_run_page_validate_zeroed(
						    chunk, run_ind+i);
					}
				}
			} else {
				/*
				 * The run is dirty, so all pages must be
				 * zeroed.
				 */
				arena_run_zero(chunk, run_ind, need_pages);
			}
		}

		/*
		 * Set the last element first, in case the run only contains one
		 * page (i.e. both statements set the same element).
		 */
		arena_mapbits_large_set(chunk, run_ind+need_pages-1, 0,
		    flag_dirty);
		arena_mapbits_large_set(chunk, run_ind, size, flag_dirty);
	} else {
		assert(zero == false);
		/*
		 * Propagate the dirty and unzeroed flags to the allocated
		 * small run, so that arena_dalloc_bin_run() has the ability to
		 * conditionally trim clean pages.
		 */
		arena_mapbits_small_set(chunk, run_ind, 0, binind, flag_dirty);
		/*
		 * The first page will always be dirtied during small run
		 * initialization, so a validation failure here would not
		 * actually cause an observable failure.
		 */
		if (config_debug && flag_dirty == 0 &&
		    arena_mapbits_unzeroed_get(chunk, run_ind) == 0)
			arena_run_page_validate_zeroed(chunk, run_ind);
		for (i = 1; i < need_pages - 1; i++) {
			arena_mapbits_small_set(chunk, run_ind+i, i, binind, 0);
			if (config_debug && flag_dirty == 0 &&
			    arena_mapbits_unzeroed_get(chunk, run_ind+i) == 0) {
				arena_run_page_validate_zeroed(chunk,
				    run_ind+i);
			}
		}
		arena_mapbits_small_set(chunk, run_ind+need_pages-1,
		    need_pages-1, binind, flag_dirty);
		if (config_debug && flag_dirty == 0 &&
		    arena_mapbits_unzeroed_get(chunk, run_ind+need_pages-1) ==
		    0) {
			arena_run_page_validate_zeroed(chunk,
			    run_ind+need_pages-1);
		}
	}
	VALGRIND_MAKE_MEM_UNDEFINED((void *)((uintptr_t)chunk + (run_ind <<
	    LG_PAGE)), (need_pages << LG_PAGE));
}

static arena_chunk_t *
arena_chunk_alloc(arena_t *arena)
{
	arena_chunk_t *chunk;
	size_t i;

	if (arena->spare != NULL) {
		chunk = arena->spare;
		arena->spare = NULL;

		assert(arena_mapbits_allocated_get(chunk, map_bias) == 0);
		assert(arena_mapbits_allocated_get(chunk, chunk_npages-1) == 0);
		assert(arena_mapbits_unallocated_size_get(chunk, map_bias) ==
		    arena_maxclass);
		assert(arena_mapbits_unallocated_size_get(chunk,
		    chunk_npages-1) == arena_maxclass);
		assert(arena_mapbits_dirty_get(chunk, map_bias) ==
		    arena_mapbits_dirty_get(chunk, chunk_npages-1));
	} else {
		bool zero;
		size_t unzeroed;

		zero = false;
		malloc_mutex_unlock(&arena->lock);
		chunk = (arena_chunk_t *)chunk_alloc(chunksize, chunksize,
		    false, &zero, arena->dss_prec);
		malloc_mutex_lock(&arena->lock);
		if (chunk == NULL)
			return (NULL);
		if (config_stats)
			arena->stats.mapped += chunksize;

		chunk->arena = arena;

		/*
		 * Claim that no pages are in use, since the header is merely
		 * overhead.
		 */
		chunk->ndirty = 0;

		chunk->nruns_avail = 0;
		chunk->nruns_adjac = 0;

		/*
		 * Initialize the map to contain one maximal free untouched run.
		 * Mark the pages as zeroed iff chunk_alloc() returned a zeroed
		 * chunk.
		 */
		unzeroed = zero ? 0 : CHUNK_MAP_UNZEROED;
		arena_mapbits_unallocated_set(chunk, map_bias, arena_maxclass,
		    unzeroed);
		/*
		 * There is no need to initialize the internal page map entries
		 * unless the chunk is not zeroed.
		 */
		if (zero == false) {
			for (i = map_bias+1; i < chunk_npages-1; i++)
				arena_mapbits_unzeroed_set(chunk, i, unzeroed);
		} else if (config_debug) {
			VALGRIND_MAKE_MEM_DEFINED(
			    (void *)arena_mapp_get(chunk, map_bias+1),
			    (void *)((uintptr_t)
			    arena_mapp_get(chunk, chunk_npages-1)
			    - (uintptr_t)arena_mapp_get(chunk, map_bias+1)));
			for (i = map_bias+1; i < chunk_npages-1; i++) {
				assert(arena_mapbits_unzeroed_get(chunk, i) ==
				    unzeroed);
			}
		}
		arena_mapbits_unallocated_set(chunk, chunk_npages-1,
		    arena_maxclass, unzeroed);
	}

	/* Insert the run into the runs_avail tree. */
	arena_avail_insert(arena, chunk, map_bias, chunk_npages-map_bias,
	    false, false);

	return (chunk);
}

static void
arena_chunk_dealloc(arena_t *arena, arena_chunk_t *chunk)
{
	assert(arena_mapbits_allocated_get(chunk, map_bias) == 0);
	assert(arena_mapbits_allocated_get(chunk, chunk_npages-1) == 0);
	assert(arena_mapbits_unallocated_size_get(chunk, map_bias) ==
	    arena_maxclass);
	assert(arena_mapbits_unallocated_size_get(chunk, chunk_npages-1) ==
	    arena_maxclass);
	assert(arena_mapbits_dirty_get(chunk, map_bias) ==
	    arena_mapbits_dirty_get(chunk, chunk_npages-1));

	/*
	 * Remove run from the runs_avail tree, so that the arena does not use
	 * it.
	 */
	arena_avail_remove(arena, chunk, map_bias, chunk_npages-map_bias,
	    false, false);

	if (arena->spare != NULL) {
		arena_chunk_t *spare = arena->spare;

		arena->spare = chunk;
		malloc_mutex_unlock(&arena->lock);
		chunk_dealloc((void *)spare, chunksize, true);
		malloc_mutex_lock(&arena->lock);
		if (config_stats)
			arena->stats.mapped -= chunksize;
	} else
		arena->spare = chunk;
}

static arena_run_t *
arena_run_alloc_helper(arena_t *arena, size_t size, bool large, size_t binind,
    bool zero)
{
	arena_run_t *run;
	arena_chunk_map_t *mapelm, key;

	key.bits = size | CHUNK_MAP_KEY;
	mapelm = arena_avail_tree_nsearch(&arena->runs_avail, &key);
	if (mapelm != NULL) {
		arena_chunk_t *run_chunk = CHUNK_ADDR2BASE(mapelm);
		size_t pageind = (((uintptr_t)mapelm -
		    (uintptr_t)run_chunk->map) / sizeof(arena_chunk_map_t))
		    + map_bias;

		run = (arena_run_t *)((uintptr_t)run_chunk + (pageind <<
		    LG_PAGE));
		arena_run_split(arena, run, size, large, binind, zero);
		return (run);
	}

	return (NULL);
}

static arena_run_t *
arena_run_alloc(arena_t *arena, size_t size, bool large, size_t binind,
    bool zero)
{
	arena_chunk_t *chunk;
	arena_run_t *run;

	assert(size <= arena_maxclass);
	assert((size & PAGE_MASK) == 0);
	assert((large && binind == BININD_INVALID) || (large == false && binind
	    != BININD_INVALID));

	/* Search the arena's chunks for the lowest best fit. */
	run = arena_run_alloc_helper(arena, size, large, binind, zero);
	if (run != NULL)
		return (run);

	/*
	 * No usable runs.  Create a new chunk from which to allocate the run.
	 */
	chunk = arena_chunk_alloc(arena);
	if (chunk != NULL) {
		run = (arena_run_t *)((uintptr_t)chunk + (map_bias << LG_PAGE));
		arena_run_split(arena, run, size, large, binind, zero);
		return (run);
	}

	/*
	 * arena_chunk_alloc() failed, but another thread may have made
	 * sufficient memory available while this one dropped arena->lock in
	 * arena_chunk_alloc(), so search one more time.
	 */
	return (arena_run_alloc_helper(arena, size, large, binind, zero));
}

static inline void
arena_maybe_purge(arena_t *arena)
{
	size_t npurgeable, threshold;

	/* Don't purge if the option is disabled. */
	if (opt_lg_dirty_mult < 0)
		return;
	/* Don't purge if all dirty pages are already being purged. */
	if (arena->ndirty <= arena->npurgatory)
		return;
	npurgeable = arena->ndirty - arena->npurgatory;
	threshold = (arena->nactive >> opt_lg_dirty_mult);
	/*
	 * Don't purge unless the number of purgeable pages exceeds the
	 * threshold.
	 */
	if (npurgeable <= threshold)
		return;

	arena_purge(arena, false);
}

static inline size_t
arena_chunk_purge(arena_t *arena, arena_chunk_t *chunk, bool all)
{
	size_t npurged;
	ql_head(arena_chunk_map_t) mapelms;
	arena_chunk_map_t *mapelm;
	size_t pageind, npages;
	size_t nmadvise;

	ql_new(&mapelms);

	/*
	 * If chunk is the spare, temporarily re-allocate it, 1) so that its
	 * run is reinserted into runs_avail, and 2) so that it cannot be
	 * completely discarded by another thread while arena->lock is dropped
	 * by this thread.  Note that the arena_run_dalloc() call will
	 * implicitly deallocate the chunk, so no explicit action is required
	 * in this function to deallocate the chunk.
	 *
	 * Note that once a chunk contains dirty pages, it cannot again contain
	 * a single run unless 1) it is a dirty run, or 2) this function purges
	 * dirty pages and causes the transition to a single clean run.  Thus
	 * (chunk == arena->spare) is possible, but it is not possible for
	 * this function to be called on the spare unless it contains a dirty
	 * run.
	 */
	if (chunk == arena->spare) {
		assert(arena_mapbits_dirty_get(chunk, map_bias) != 0);
		assert(arena_mapbits_dirty_get(chunk, chunk_npages-1) != 0);

		arena_chunk_alloc(arena);
	}

	if (config_stats)
		arena->stats.purged += chunk->ndirty;

	/*
	 * Operate on all dirty runs if there is no clean/dirty run
	 * fragmentation.
	 */
	if (chunk->nruns_adjac == 0)
		all = true;

	/*
	 * Temporarily allocate free dirty runs within chunk.  If all is false,
	 * only operate on dirty runs that are fragments; otherwise operate on
	 * all dirty runs.
	 */
	for (pageind = map_bias; pageind < chunk_npages; pageind += npages) {
		mapelm = arena_mapp_get(chunk, pageind);
		if (arena_mapbits_allocated_get(chunk, pageind) == 0) {
			size_t run_size =
			    arena_mapbits_unallocated_size_get(chunk, pageind);

			npages = run_size >> LG_PAGE;
			assert(pageind + npages <= chunk_npages);
			assert(arena_mapbits_dirty_get(chunk, pageind) ==
			    arena_mapbits_dirty_get(chunk, pageind+npages-1));

			if (arena_mapbits_dirty_get(chunk, pageind) != 0 &&
			    (all || arena_avail_adjac(chunk, pageind,
			    npages))) {
				arena_run_t *run = (arena_run_t *)((uintptr_t)
				    chunk + (uintptr_t)(pageind << LG_PAGE));

				arena_run_split(arena, run, run_size, true,
				    BININD_INVALID, false);
				/* Append to list for later processing. */
				ql_elm_new(mapelm, u.ql_link);
				ql_tail_insert(&mapelms, mapelm, u.ql_link);
			}
		} else {
			/* Skip run. */
			if (arena_mapbits_large_get(chunk, pageind) != 0) {
				npages = arena_mapbits_large_size_get(chunk,
				    pageind) >> LG_PAGE;
			} else {
				size_t binind;
				arena_bin_info_t *bin_info;
				arena_run_t *run = (arena_run_t *)((uintptr_t)
				    chunk + (uintptr_t)(pageind << LG_PAGE));

				assert(arena_mapbits_small_runind_get(chunk,
				    pageind) == 0);
				binind = arena_bin_index(arena, run->bin);
				bin_info = &arena_bin_info[binind];
				npages = bin_info->run_size >> LG_PAGE;
			}
		}
	}
	assert(pageind == chunk_npages);
	assert(chunk->ndirty == 0 || all == false);
	assert(chunk->nruns_adjac == 0);

	malloc_mutex_unlock(&arena->lock);
	if (config_stats)
		nmadvise = 0;
	npurged = 0;
	ql_foreach(mapelm, &mapelms, u.ql_link) {
		bool unzeroed;
		size_t flag_unzeroed, i;

		pageind = (((uintptr_t)mapelm - (uintptr_t)chunk->map) /
		    sizeof(arena_chunk_map_t)) + map_bias;
		npages = arena_mapbits_large_size_get(chunk, pageind) >>
		    LG_PAGE;
		assert(pageind + npages <= chunk_npages);
		unzeroed = pages_purge((void *)((uintptr_t)chunk + (pageind <<
		    LG_PAGE)), (npages << LG_PAGE));
		flag_unzeroed = unzeroed ? CHUNK_MAP_UNZEROED : 0;
		/*
		 * Set the unzeroed flag for all pages, now that pages_purge()
		 * has returned whether the pages were zeroed as a side effect
		 * of purging.  This chunk map modification is safe even though
		 * the arena mutex isn't currently owned by this thread,
		 * because the run is marked as allocated, thus protecting it
		 * from being modified by any other thread.  As long as these
		 * writes don't perturb the first and last elements'
		 * CHUNK_MAP_ALLOCATED bits, behavior is well defined.
		 */
		for (i = 0; i < npages; i++) {
			arena_mapbits_unzeroed_set(chunk, pageind+i,
			    flag_unzeroed);
		}
		npurged += npages;
		if (config_stats)
			nmadvise++;
	}
	malloc_mutex_lock(&arena->lock);
	if (config_stats)
		arena->stats.nmadvise += nmadvise;

	/* Deallocate runs. */
	for (mapelm = ql_first(&mapelms); mapelm != NULL;
	    mapelm = ql_first(&mapelms)) {
		arena_run_t *run;

		pageind = (((uintptr_t)mapelm - (uintptr_t)chunk->map) /
		    sizeof(arena_chunk_map_t)) + map_bias;
		run = (arena_run_t *)((uintptr_t)chunk + (uintptr_t)(pageind <<
		    LG_PAGE));
		ql_remove(&mapelms, mapelm, u.ql_link);
		arena_run_dalloc(arena, run, false, true);
	}

	return (npurged);
}

static arena_chunk_t *
chunks_dirty_iter_cb(arena_chunk_tree_t *tree, arena_chunk_t *chunk, void *arg)
{
       size_t *ndirty = (size_t *)arg;

       assert(chunk->ndirty != 0);
       *ndirty += chunk->ndirty;
       return (NULL);
}

static void
arena_purge(arena_t *arena, bool all)
{
	arena_chunk_t *chunk;
	size_t npurgatory;
	if (config_debug) {
		size_t ndirty = 0;

		arena_chunk_dirty_iter(&arena->chunks_dirty, NULL,
		    chunks_dirty_iter_cb, (void *)&ndirty);
		assert(ndirty == arena->ndirty);
	}
	assert(arena->ndirty > arena->npurgatory || all);
	assert((arena->nactive >> opt_lg_dirty_mult) < (arena->ndirty -
	    arena->npurgatory) || all);

	if (config_stats)
		arena->stats.npurge++;

	/*
	 * Compute the minimum number of pages that this thread should try to
	 * purge, and add the result to arena->npurgatory.  This will keep
	 * multiple threads from racing to reduce ndirty below the threshold.
	 */
	{
		size_t npurgeable = arena->ndirty - arena->npurgatory;

		if (all == false) {
			size_t threshold = (arena->nactive >>
			    opt_lg_dirty_mult);

			npurgatory = npurgeable - threshold;
		} else
			npurgatory = npurgeable;
	}
	arena->npurgatory += npurgatory;

	while (npurgatory > 0) {
		size_t npurgeable, npurged, nunpurged;

		/* Get next chunk with dirty pages. */
		chunk = arena_chunk_dirty_first(&arena->chunks_dirty);
		if (chunk == NULL) {
			/*
			 * This thread was unable to purge as many pages as
			 * originally intended, due to races with other threads
			 * that either did some of the purging work, or re-used
			 * dirty pages.
			 */
			arena->npurgatory -= npurgatory;
			return;
		}
		npurgeable = chunk->ndirty;
		assert(npurgeable != 0);

		if (npurgeable > npurgatory && chunk->nruns_adjac == 0) {
			/*
			 * This thread will purge all the dirty pages in chunk,
			 * so set npurgatory to reflect this thread's intent to
			 * purge the pages.  This tends to reduce the chances
			 * of the following scenario:
			 *
			 * 1) This thread sets arena->npurgatory such that
			 *    (arena->ndirty - arena->npurgatory) is at the
			 *    threshold.
			 * 2) This thread drops arena->lock.
			 * 3) Another thread causes one or more pages to be
			 *    dirtied, and immediately determines that it must
			 *    purge dirty pages.
			 *
			 * If this scenario *does* play out, that's okay,
			 * because all of the purging work being done really
			 * needs to happen.
			 */
			arena->npurgatory += npurgeable - npurgatory;
			npurgatory = npurgeable;
		}

		/*
		 * Keep track of how many pages are purgeable, versus how many
		 * actually get purged, and adjust counters accordingly.
		 */
		arena->npurgatory -= npurgeable;
		npurgatory -= npurgeable;
		npurged = arena_chunk_purge(arena, chunk, all);
		nunpurged = npurgeable - npurged;
		arena->npurgatory += nunpurged;
		npurgatory += nunpurged;
	}
}

void
arena_purge_all(arena_t *arena)
{

	malloc_mutex_lock(&arena->lock);
	arena_purge(arena, true);
	malloc_mutex_unlock(&arena->lock);
}

static void
arena_run_dalloc(arena_t *arena, arena_run_t *run, bool dirty, bool cleaned)
{
	arena_chunk_t *chunk;
	size_t size, run_ind, run_pages, flag_dirty;

	chunk = (arena_chunk_t *)CHUNK_ADDR2BASE(run);
	run_ind = (size_t)(((uintptr_t)run - (uintptr_t)chunk) >> LG_PAGE);
	assert(run_ind >= map_bias);
	assert(run_ind < chunk_npages);
	if (arena_mapbits_large_get(chunk, run_ind) != 0) {
		size = arena_mapbits_large_size_get(chunk, run_ind);
		assert(size == PAGE ||
		    arena_mapbits_large_size_get(chunk,
		    run_ind+(size>>LG_PAGE)-1) == 0);
	} else {
		size_t binind = arena_bin_index(arena, run->bin);
		arena_bin_info_t *bin_info = &arena_bin_info[binind];
		size = bin_info->run_size;
	}
	run_pages = (size >> LG_PAGE);
	if (config_stats) {
		/*
		 * Update stats_cactive if nactive is crossing a chunk
		 * multiple.
		 */
		size_t cactive_diff = CHUNK_CEILING(arena->nactive << LG_PAGE) -
		    CHUNK_CEILING((arena->nactive - run_pages) << LG_PAGE);
		if (cactive_diff != 0)
			stats_cactive_sub(cactive_diff);
	}
	arena->nactive -= run_pages;

	/*
	 * The run is dirty if the caller claims to have dirtied it, as well as
	 * if it was already dirty before being allocated and the caller
	 * doesn't claim to have cleaned it.
	 */
	assert(arena_mapbits_dirty_get(chunk, run_ind) ==
	    arena_mapbits_dirty_get(chunk, run_ind+run_pages-1));
	if (cleaned == false && arena_mapbits_dirty_get(chunk, run_ind) != 0)
		dirty = true;
	flag_dirty = dirty ? CHUNK_MAP_DIRTY : 0;

	/* Mark pages as unallocated in the chunk map. */
	if (dirty) {
		arena_mapbits_unallocated_set(chunk, run_ind, size,
		    CHUNK_MAP_DIRTY);
		arena_mapbits_unallocated_set(chunk, run_ind+run_pages-1, size,
		    CHUNK_MAP_DIRTY);
	} else {
		arena_mapbits_unallocated_set(chunk, run_ind, size,
		    arena_mapbits_unzeroed_get(chunk, run_ind));
		arena_mapbits_unallocated_set(chunk, run_ind+run_pages-1, size,
		    arena_mapbits_unzeroed_get(chunk, run_ind+run_pages-1));
	}

	/* Try to coalesce forward. */
	if (run_ind + run_pages < chunk_npages &&
	    arena_mapbits_allocated_get(chunk, run_ind+run_pages) == 0 &&
	    arena_mapbits_dirty_get(chunk, run_ind+run_pages) == flag_dirty) {
		size_t nrun_size = arena_mapbits_unallocated_size_get(chunk,
		    run_ind+run_pages);
		size_t nrun_pages = nrun_size >> LG_PAGE;

		/*
		 * Remove successor from runs_avail; the coalesced run is
		 * inserted later.
		 */
		assert(arena_mapbits_unallocated_size_get(chunk,
		    run_ind+run_pages+nrun_pages-1) == nrun_size);
		assert(arena_mapbits_dirty_get(chunk,
		    run_ind+run_pages+nrun_pages-1) == flag_dirty);
		arena_avail_remove(arena, chunk, run_ind+run_pages, nrun_pages,
		    false, true);

		size += nrun_size;
		run_pages += nrun_pages;

		arena_mapbits_unallocated_size_set(chunk, run_ind, size);
		arena_mapbits_unallocated_size_set(chunk, run_ind+run_pages-1,
		    size);
	}

	/* Try to coalesce backward. */
	if (run_ind > map_bias && arena_mapbits_allocated_get(chunk, run_ind-1)
	    == 0 && arena_mapbits_dirty_get(chunk, run_ind-1) == flag_dirty) {
		size_t prun_size = arena_mapbits_unallocated_size_get(chunk,
		    run_ind-1);
		size_t prun_pages = prun_size >> LG_PAGE;

		run_ind -= prun_pages;

		/*
		 * Remove predecessor from runs_avail; the coalesced run is
		 * inserted later.
		 */
		assert(arena_mapbits_unallocated_size_get(chunk, run_ind) ==
		    prun_size);
		assert(arena_mapbits_dirty_get(chunk, run_ind) == flag_dirty);
		arena_avail_remove(arena, chunk, run_ind, prun_pages, true,
		    false);

		size += prun_size;
		run_pages += prun_pages;

		arena_mapbits_unallocated_size_set(chunk, run_ind, size);
		arena_mapbits_unallocated_size_set(chunk, run_ind+run_pages-1,
		    size);
	}

	/* Insert into runs_avail, now that coalescing is complete. */
	assert(arena_mapbits_unallocated_size_get(chunk, run_ind) ==
	    arena_mapbits_unallocated_size_get(chunk, run_ind+run_pages-1));
	assert(arena_mapbits_dirty_get(chunk, run_ind) ==
	    arena_mapbits_dirty_get(chunk, run_ind+run_pages-1));
	arena_avail_insert(arena, chunk, run_ind, run_pages, true, true);

	/* Deallocate chunk if it is now completely unused. */
	if (size == arena_maxclass) {
		assert(run_ind == map_bias);
		assert(run_pages == (arena_maxclass >> LG_PAGE));
		arena_chunk_dealloc(arena, chunk);
	}

	/*
	 * It is okay to do dirty page processing here even if the chunk was
	 * deallocated above, since in that case it is the spare.  Waiting
	 * until after possible chunk deallocation to do dirty processing
	 * allows for an old spare to be fully deallocated, thus decreasing the
	 * chances of spuriously crossing the dirty page purging threshold.
	 */
	if (dirty)
		arena_maybe_purge(arena);
}

static void
arena_run_trim_head(arena_t *arena, arena_chunk_t *chunk, arena_run_t *run,
    size_t oldsize, size_t newsize)
{
	size_t pageind = ((uintptr_t)run - (uintptr_t)chunk) >> LG_PAGE;
	size_t head_npages = (oldsize - newsize) >> LG_PAGE;
	size_t flag_dirty = arena_mapbits_dirty_get(chunk, pageind);

	assert(oldsize > newsize);

	/*
	 * Update the chunk map so that arena_run_dalloc() can treat the
	 * leading run as separately allocated.  Set the last element of each
	 * run first, in case of single-page runs.
	 */
	assert(arena_mapbits_large_size_get(chunk, pageind) == oldsize);
	arena_mapbits_large_set(chunk, pageind+head_npages-1, 0, flag_dirty);
	arena_mapbits_large_set(chunk, pageind, oldsize-newsize, flag_dirty);

	if (config_debug) {
		UNUSED size_t tail_npages = newsize >> LG_PAGE;
		assert(arena_mapbits_large_size_get(chunk,
		    pageind+head_npages+tail_npages-1) == 0);
		assert(arena_mapbits_dirty_get(chunk,
		    pageind+head_npages+tail_npages-1) == flag_dirty);
	}
	arena_mapbits_large_set(chunk, pageind+head_npages, newsize,
	    flag_dirty);

	arena_run_dalloc(arena, run, false, false);
}

static void
arena_run_trim_tail(arena_t *arena, arena_chunk_t *chunk, arena_run_t *run,
    size_t oldsize, size_t newsize, bool dirty)
{
	size_t pageind = ((uintptr_t)run - (uintptr_t)chunk) >> LG_PAGE;
	size_t head_npages = newsize >> LG_PAGE;
	size_t flag_dirty = arena_mapbits_dirty_get(chunk, pageind);

	assert(oldsize > newsize);

	/*
	 * Update the chunk map so that arena_run_dalloc() can treat the
	 * trailing run as separately allocated.  Set the last element of each
	 * run first, in case of single-page runs.
	 */
	assert(arena_mapbits_large_size_get(chunk, pageind) == oldsize);
	arena_mapbits_large_set(chunk, pageind+head_npages-1, 0, flag_dirty);
	arena_mapbits_large_set(chunk, pageind, newsize, flag_dirty);

	if (config_debug) {
		UNUSED size_t tail_npages = (oldsize - newsize) >> LG_PAGE;
		assert(arena_mapbits_large_size_get(chunk,
		    pageind+head_npages+tail_npages-1) == 0);
		assert(arena_mapbits_dirty_get(chunk,
		    pageind+head_npages+tail_npages-1) == flag_dirty);
	}
	arena_mapbits_large_set(chunk, pageind+head_npages, oldsize-newsize,
	    flag_dirty);

	arena_run_dalloc(arena, (arena_run_t *)((uintptr_t)run + newsize),
	    dirty, false);
}

static arena_run_t *
arena_bin_runs_first(arena_bin_t *bin)
{
	arena_chunk_map_t *mapelm = arena_run_tree_first(&bin->runs);
	if (mapelm != NULL) {
		arena_chunk_t *chunk;
		size_t pageind;
		arena_run_t *run;

		chunk = (arena_chunk_t *)CHUNK_ADDR2BASE(mapelm);
		pageind = ((((uintptr_t)mapelm - (uintptr_t)chunk->map) /
		    sizeof(arena_chunk_map_t))) + map_bias;
		run = (arena_run_t *)((uintptr_t)chunk + (uintptr_t)((pageind -
		    arena_mapbits_small_runind_get(chunk, pageind)) <<
		    LG_PAGE));
		return (run);
	}

	return (NULL);
}

static void
arena_bin_runs_insert(arena_bin_t *bin, arena_run_t *run)
{
	arena_chunk_t *chunk = CHUNK_ADDR2BASE(run);
	size_t pageind = ((uintptr_t)run - (uintptr_t)chunk) >> LG_PAGE;
	arena_chunk_map_t *mapelm = arena_mapp_get(chunk, pageind);

	assert(arena_run_tree_search(&bin->runs, mapelm) == NULL);

	arena_run_tree_insert(&bin->runs, mapelm);
}

static void
arena_bin_runs_remove(arena_bin_t *bin, arena_run_t *run)
{
	arena_chunk_t *chunk = (arena_chunk_t *)CHUNK_ADDR2BASE(run);
	size_t pageind = ((uintptr_t)run - (uintptr_t)chunk) >> LG_PAGE;
	arena_chunk_map_t *mapelm = arena_mapp_get(chunk, pageind);

	assert(arena_run_tree_search(&bin->runs, mapelm) != NULL);

	arena_run_tree_remove(&bin->runs, mapelm);
}

static arena_run_t *
arena_bin_nonfull_run_tryget(arena_bin_t *bin)
{
	arena_run_t *run = arena_bin_runs_first(bin);
	if (run != NULL) {
		arena_bin_runs_remove(bin, run);
		if (config_stats)
			bin->stats.reruns++;
	}
	return (run);
}

static arena_run_t *
arena_bin_nonfull_run_get(arena_t *arena, arena_bin_t *bin)
{
	arena_run_t *run;
	size_t binind;
	arena_bin_info_t *bin_info;

	/* Look for a usable run. */
	run = arena_bin_nonfull_run_tryget(bin);
	if (run != NULL)
		return (run);
	/* No existing runs have any space available. */

	binind = arena_bin_index(arena, bin);
	bin_info = &arena_bin_info[binind];

	/* Allocate a new run. */
	malloc_mutex_unlock(&bin->lock);
	/******************************/
	malloc_mutex_lock(&arena->lock);
	run = arena_run_alloc(arena, bin_info->run_size, false, binind, false);
	if (run != NULL) {
		bitmap_t *bitmap = (bitmap_t *)((uintptr_t)run +
		    (uintptr_t)bin_info->bitmap_offset);

		/* Initialize run internals. */
		run->bin = bin;
		run->nextind = 0;
		run->nfree = bin_info->nregs;
		bitmap_init(bitmap, &bin_info->bitmap_info);
	}
	malloc_mutex_unlock(&arena->lock);
	/********************************/
	malloc_mutex_lock(&bin->lock);
	if (run != NULL) {
		if (config_stats) {
			bin->stats.nruns++;
			bin->stats.curruns++;
		}
		return (run);
	}

	/*
	 * arena_run_alloc() failed, but another thread may have made
	 * sufficient memory available while this one dropped bin->lock above,
	 * so search one more time.
	 */
	run = arena_bin_nonfull_run_tryget(bin);
	if (run != NULL)
		return (run);

	return (NULL);
}

/* Re-fill bin->runcur, then call arena_run_reg_alloc(). */
static void *
arena_bin_malloc_hard(arena_t *arena, arena_bin_t *bin)
{
	void *ret;
	size_t binind;
	arena_bin_info_t *bin_info;
	arena_run_t *run;

	binind = arena_bin_index(arena, bin);
	bin_info = &arena_bin_info[binind];
	bin->runcur = NULL;
	run = arena_bin_nonfull_run_get(arena, bin);
	if (bin->runcur != NULL && bin->runcur->nfree > 0) {
		/*
		 * Another thread updated runcur while this one ran without the
		 * bin lock in arena_bin_nonfull_run_get().
		 */
		assert(bin->runcur->nfree > 0);
		ret = arena_run_reg_alloc(bin->runcur, bin_info);
		if (run != NULL) {
			arena_chunk_t *chunk;

			/*
			 * arena_run_alloc() may have allocated run, or it may
			 * have pulled run from the bin's run tree.  Therefore
			 * it is unsafe to make any assumptions about how run
			 * has previously been used, and arena_bin_lower_run()
			 * must be called, as if a region were just deallocated
			 * from the run.
			 */
			chunk = (arena_chunk_t *)CHUNK_ADDR2BASE(run);
			if (run->nfree == bin_info->nregs)
				arena_dalloc_bin_run(arena, chunk, run, bin);
			else
				arena_bin_lower_run(arena, chunk, run, bin);
		}
		return (ret);
	}

	if (run == NULL)
		return (NULL);

	bin->runcur = run;

	assert(bin->runcur->nfree > 0);

	return (arena_run_reg_alloc(bin->runcur, bin_info));
}

void
arena_tcache_fill_small(arena_t *arena, tcache_bin_t *tbin, size_t binind,
    uint64_t prof_accumbytes)
{
	unsigned i, nfill;
	arena_bin_t *bin;
	arena_run_t *run;
	void *ptr;

	assert(tbin->ncached == 0);

	if (config_prof && arena_prof_accum(arena, prof_accumbytes))
		prof_idump();
	bin = &arena->bins[binind];
	malloc_mutex_lock(&bin->lock);
	for (i = 0, nfill = (tcache_bin_info[binind].ncached_max >>
	    tbin->lg_fill_div); i < nfill; i++) {
		if ((run = bin->runcur) != NULL && run->nfree > 0)
			ptr = arena_run_reg_alloc(run, &arena_bin_info[binind]);
		else
			ptr = arena_bin_malloc_hard(arena, bin);
		if (ptr == NULL)
			break;
		if (config_fill && opt_junk) {
			arena_alloc_junk_small(ptr, &arena_bin_info[binind],
			    true);
		}
		/* Insert such that low regions get used first. */
		tbin->avail[nfill - 1 - i] = ptr;
	}
	if (config_stats) {
		bin->stats.allocated += i * arena_bin_info[binind].reg_size;
		bin->stats.nmalloc += i;
		bin->stats.nrequests += tbin->tstats.nrequests;
		bin->stats.nfills++;
		tbin->tstats.nrequests = 0;
	}
	malloc_mutex_unlock(&bin->lock);
	tbin->ncached = i;
}

void
arena_alloc_junk_small(void *ptr, arena_bin_info_t *bin_info, bool zero)
{

	if (zero) {
		size_t redzone_size = bin_info->redzone_size;
		memset((void *)((uintptr_t)ptr - redzone_size), 0xa5,
		    redzone_size);
		memset((void *)((uintptr_t)ptr + bin_info->reg_size), 0xa5,
		    redzone_size);
	} else {
		memset((void *)((uintptr_t)ptr - bin_info->redzone_size), 0xa5,
		    bin_info->reg_interval);
	}
}

void
arena_dalloc_junk_small(void *ptr, arena_bin_info_t *bin_info)
{
	size_t size = bin_info->reg_size;
	size_t redzone_size = bin_info->redzone_size;
	size_t i;
	bool error = false;

	for (i = 1; i <= redzone_size; i++) {
		unsigned byte;
		if ((byte = *(uint8_t *)((uintptr_t)ptr - i)) != 0xa5) {
			error = true;
			malloc_printf("<jemalloc>: Corrupt redzone "
			    "%zu byte%s before %p (size %zu), byte=%#x\n", i,
			    (i == 1) ? "" : "s", ptr, size, byte);
		}
	}
	for (i = 0; i < redzone_size; i++) {
		unsigned byte;
		if ((byte = *(uint8_t *)((uintptr_t)ptr + size + i)) != 0xa5) {
			error = true;
			malloc_printf("<jemalloc>: Corrupt redzone "
			    "%zu byte%s after end of %p (size %zu), byte=%#x\n",
			    i, (i == 1) ? "" : "s", ptr, size, byte);
		}
	}
	if (opt_abort && error)
		abort();

	memset((void *)((uintptr_t)ptr - redzone_size), 0x5a,
	    bin_info->reg_interval);
}

void *
arena_malloc_small(arena_t *arena, size_t size, bool zero)
{
	void *ret;
	arena_bin_t *bin;
	arena_run_t *run;
	size_t binind;

	binind = SMALL_SIZE2BIN(size);
	assert(binind < NBINS);
	bin = &arena->bins[binind];
	size = arena_bin_info[binind].reg_size;

	malloc_mutex_lock(&bin->lock);
	if ((run = bin->runcur) != NULL && run->nfree > 0)
		ret = arena_run_reg_alloc(run, &arena_bin_info[binind]);
	else
		ret = arena_bin_malloc_hard(arena, bin);

	if (ret == NULL) {
		malloc_mutex_unlock(&bin->lock);
		return (NULL);
	}

	if (config_stats) {
		bin->stats.allocated += size;
		bin->stats.nmalloc++;
		bin->stats.nrequests++;
	}
	malloc_mutex_unlock(&bin->lock);
	if (config_prof && isthreaded == false && arena_prof_accum(arena, size))
		prof_idump();

	if (zero == false) {
		if (config_fill) {
			if (opt_junk) {
				arena_alloc_junk_small(ret,
				    &arena_bin_info[binind], false);
			} else if (opt_zero)
				memset(ret, 0, size);
		}
	} else {
		if (config_fill && opt_junk) {
			arena_alloc_junk_small(ret, &arena_bin_info[binind],
			    true);
		}
		VALGRIND_MAKE_MEM_UNDEFINED(ret, size);
		memset(ret, 0, size);
	}
	VALGRIND_MAKE_MEM_UNDEFINED(ret, size);

	return (ret);
}

void *
arena_malloc_large(arena_t *arena, size_t size, bool zero)
{
	void *ret;
	UNUSED bool idump;

	/* Large allocation. */
	size = PAGE_CEILING(size);
	malloc_mutex_lock(&arena->lock);
	ret = (void *)arena_run_alloc(arena, size, true, BININD_INVALID, zero);
	if (ret == NULL) {
		malloc_mutex_unlock(&arena->lock);
		return (NULL);
	}
	if (config_stats) {
		arena->stats.nmalloc_large++;
		arena->stats.nrequests_large++;
		arena->stats.allocated_large += size;
		arena->stats.lstats[(size >> LG_PAGE) - 1].nmalloc++;
		arena->stats.lstats[(size >> LG_PAGE) - 1].nrequests++;
		arena->stats.lstats[(size >> LG_PAGE) - 1].curruns++;
	}
	if (config_prof)
		idump = arena_prof_accum_locked(arena, size);
	malloc_mutex_unlock(&arena->lock);
	if (config_prof && idump)
		prof_idump();

	if (zero == false) {
		if (config_fill) {
			if (opt_junk)
				memset(ret, 0xa5, size);
			else if (opt_zero)
				memset(ret, 0, size);
		}
	}

	return (ret);
}

/* Only handles large allocations that require more than page alignment. */
void *
arena_palloc(arena_t *arena, size_t size, size_t alignment, bool zero)
{
	void *ret;
	size_t alloc_size, leadsize, trailsize;
	arena_run_t *run;
	arena_chunk_t *chunk;

	assert((size & PAGE_MASK) == 0);

	alignment = PAGE_CEILING(alignment);
	alloc_size = size + alignment - PAGE;

	malloc_mutex_lock(&arena->lock);
	run = arena_run_alloc(arena, alloc_size, true, BININD_INVALID, zero);
	if (run == NULL) {
		malloc_mutex_unlock(&arena->lock);
		return (NULL);
	}
	chunk = (arena_chunk_t *)CHUNK_ADDR2BASE(run);

	leadsize = ALIGNMENT_CEILING((uintptr_t)run, alignment) -
	    (uintptr_t)run;
	assert(alloc_size >= leadsize + size);
	trailsize = alloc_size - leadsize - size;
	ret = (void *)((uintptr_t)run + leadsize);
	if (leadsize != 0) {
		arena_run_trim_head(arena, chunk, run, alloc_size, alloc_size -
		    leadsize);
	}
	if (trailsize != 0) {
		arena_run_trim_tail(arena, chunk, ret, size + trailsize, size,
		    false);
	}

	if (config_stats) {
		arena->stats.nmalloc_large++;
		arena->stats.nrequests_large++;
		arena->stats.allocated_large += size;
		arena->stats.lstats[(size >> LG_PAGE) - 1].nmalloc++;
		arena->stats.lstats[(size >> LG_PAGE) - 1].nrequests++;
		arena->stats.lstats[(size >> LG_PAGE) - 1].curruns++;
	}
	malloc_mutex_unlock(&arena->lock);

	if (config_fill && zero == false) {
		if (opt_junk)
			memset(ret, 0xa5, size);
		else if (opt_zero)
			memset(ret, 0, size);
	}
	return (ret);
}

void
arena_prof_promoted(const void *ptr, size_t size)
{
	arena_chunk_t *chunk;
	size_t pageind, binind;

	cassert(config_prof);
	assert(ptr != NULL);
	assert(CHUNK_ADDR2BASE(ptr) != ptr);
	assert(isalloc(ptr, false) == PAGE);
	assert(isalloc(ptr, true) == PAGE);
	assert(size <= SMALL_MAXCLASS);

	chunk = (arena_chunk_t *)CHUNK_ADDR2BASE(ptr);
	pageind = ((uintptr_t)ptr - (uintptr_t)chunk) >> LG_PAGE;
	binind = SMALL_SIZE2BIN(size);
	assert(binind < NBINS);
	arena_mapbits_large_binind_set(chunk, pageind, binind);

	assert(isalloc(ptr, false) == PAGE);
	assert(isalloc(ptr, true) == size);
}

static void
arena_dissociate_bin_run(arena_chunk_t *chunk, arena_run_t *run,
    arena_bin_t *bin)
{

	/* Dissociate run from bin. */
	if (run == bin->runcur)
		bin->runcur = NULL;
	else {
		size_t binind = arena_bin_index(chunk->arena, bin);
		arena_bin_info_t *bin_info = &arena_bin_info[binind];

		if (bin_info->nregs != 1) {
			/*
			 * This block's conditional is necessary because if the
			 * run only contains one region, then it never gets
			 * inserted into the non-full runs tree.
			 */
			arena_bin_runs_remove(bin, run);
		}
	}
}

static void
arena_dalloc_bin_run(arena_t *arena, arena_chunk_t *chunk, arena_run_t *run,
    arena_bin_t *bin)
{
	size_t binind;
	arena_bin_info_t *bin_info;
	size_t npages, run_ind, past;

	assert(run != bin->runcur);
	assert(arena_run_tree_search(&bin->runs,
	    arena_mapp_get(chunk, ((uintptr_t)run-(uintptr_t)chunk)>>LG_PAGE))
	    == NULL);

	binind = arena_bin_index(chunk->arena, run->bin);
	bin_info = &arena_bin_info[binind];

	malloc_mutex_unlock(&bin->lock);
	/******************************/
	npages = bin_info->run_size >> LG_PAGE;
	run_ind = (size_t)(((uintptr_t)run - (uintptr_t)chunk) >> LG_PAGE);
	past = (size_t)(PAGE_CEILING((uintptr_t)run +
	    (uintptr_t)bin_info->reg0_offset + (uintptr_t)(run->nextind *
	    bin_info->reg_interval - bin_info->redzone_size) -
	    (uintptr_t)chunk) >> LG_PAGE);
	malloc_mutex_lock(&arena->lock);

	/*
	 * If the run was originally clean, and some pages were never touched,
	 * trim the clean pages before deallocating the dirty portion of the
	 * run.
	 */
	assert(arena_mapbits_dirty_get(chunk, run_ind) ==
	    arena_mapbits_dirty_get(chunk, run_ind+npages-1));
	if (arena_mapbits_dirty_get(chunk, run_ind) == 0 && past - run_ind <
	    npages) {
		/* Trim clean pages.  Convert to large run beforehand. */
		assert(npages > 0);
		arena_mapbits_large_set(chunk, run_ind, bin_info->run_size, 0);
		arena_mapbits_large_set(chunk, run_ind+npages-1, 0, 0);
		arena_run_trim_tail(arena, chunk, run, (npages << LG_PAGE),
		    ((past - run_ind) << LG_PAGE), false);
		/* npages = past - run_ind; */
	}
	arena_run_dalloc(arena, run, true, false);
	malloc_mutex_unlock(&arena->lock);
	/****************************/
	malloc_mutex_lock(&bin->lock);
	if (config_stats)
		bin->stats.curruns--;
}

static void
arena_bin_lower_run(arena_t *arena, arena_chunk_t *chunk, arena_run_t *run,
    arena_bin_t *bin)
{

	/*
	 * Make sure that if bin->runcur is non-NULL, it refers to the lowest
	 * non-full run.  It is okay to NULL runcur out rather than proactively
	 * keeping it pointing at the lowest non-full run.
	 */
	if ((uintptr_t)run < (uintptr_t)bin->runcur) {
		/* Switch runcur. */
		if (bin->runcur->nfree > 0)
			arena_bin_runs_insert(bin, bin->runcur);
		bin->runcur = run;
		if (config_stats)
			bin->stats.reruns++;
	} else
		arena_bin_runs_insert(bin, run);
}

void
arena_dalloc_bin_locked(arena_t *arena, arena_chunk_t *chunk, void *ptr,
    arena_chunk_map_t *mapelm)
{
	size_t pageind;
	arena_run_t *run;
	arena_bin_t *bin;
	arena_bin_info_t *bin_info;
	size_t size, binind;

	pageind = ((uintptr_t)ptr - (uintptr_t)chunk) >> LG_PAGE;
	run = (arena_run_t *)((uintptr_t)chunk + (uintptr_t)((pageind -
	    arena_mapbits_small_runind_get(chunk, pageind)) << LG_PAGE));
	bin = run->bin;
	binind = arena_ptr_small_binind_get(ptr, mapelm->bits);
	bin_info = &arena_bin_info[binind];
	if (config_fill || config_stats)
		size = bin_info->reg_size;

	if (config_fill && opt_junk)
		arena_dalloc_junk_small(ptr, bin_info);

	arena_run_reg_dalloc(run, ptr);
	if (run->nfree == bin_info->nregs) {
		arena_dissociate_bin_run(chunk, run, bin);
		arena_dalloc_bin_run(arena, chunk, run, bin);
	} else if (run->nfree == 1 && run != bin->runcur)
		arena_bin_lower_run(arena, chunk, run, bin);

	if (config_stats) {
		bin->stats.allocated -= size;
		bin->stats.ndalloc++;
	}
}

void
arena_dalloc_bin(arena_t *arena, arena_chunk_t *chunk, void *ptr,
    size_t pageind, arena_chunk_map_t *mapelm)
{
	arena_run_t *run;
	arena_bin_t *bin;

	run = (arena_run_t *)((uintptr_t)chunk + (uintptr_t)((pageind -
	    arena_mapbits_small_runind_get(chunk, pageind)) << LG_PAGE));
	bin = run->bin;
	malloc_mutex_lock(&bin->lock);
	arena_dalloc_bin_locked(arena, chunk, ptr, mapelm);
	malloc_mutex_unlock(&bin->lock);
}

void
arena_dalloc_small(arena_t *arena, arena_chunk_t *chunk, void *ptr,
    size_t pageind)
{
	arena_chunk_map_t *mapelm;

	if (config_debug) {
		/* arena_ptr_small_binind_get() does extra sanity checking. */
		assert(arena_ptr_small_binind_get(ptr, arena_mapbits_get(chunk,
		    pageind)) != BININD_INVALID);
	}
	mapelm = arena_mapp_get(chunk, pageind);
	arena_dalloc_bin(arena, chunk, ptr, pageind, mapelm);
}

void
arena_dalloc_large_locked(arena_t *arena, arena_chunk_t *chunk, void *ptr)
{

	if (config_fill || config_stats) {
		size_t pageind = ((uintptr_t)ptr - (uintptr_t)chunk) >> LG_PAGE;
		size_t size = arena_mapbits_large_size_get(chunk, pageind);

		if (config_fill && config_stats && opt_junk)
			memset(ptr, 0x5a, size);
		if (config_stats) {
			arena->stats.ndalloc_large++;
			arena->stats.allocated_large -= size;
			arena->stats.lstats[(size >> LG_PAGE) - 1].ndalloc++;
			arena->stats.lstats[(size >> LG_PAGE) - 1].curruns--;
		}
	}

	arena_run_dalloc(arena, (arena_run_t *)ptr, true, false);
}

void
arena_dalloc_large(arena_t *arena, arena_chunk_t *chunk, void *ptr)
{

	malloc_mutex_lock(&arena->lock);
	arena_dalloc_large_locked(arena, chunk, ptr);
	malloc_mutex_unlock(&arena->lock);
}

static void
arena_ralloc_large_shrink(arena_t *arena, arena_chunk_t *chunk, void *ptr,
    size_t oldsize, size_t size)
{

	assert(size < oldsize);

	/*
	 * Shrink the run, and make trailing pages available for other
	 * allocations.
	 */
	malloc_mutex_lock(&arena->lock);
	arena_run_trim_tail(arena, chunk, (arena_run_t *)ptr, oldsize, size,
	    true);
	if (config_stats) {
		arena->stats.ndalloc_large++;
		arena->stats.allocated_large -= oldsize;
		arena->stats.lstats[(oldsize >> LG_PAGE) - 1].ndalloc++;
		arena->stats.lstats[(oldsize >> LG_PAGE) - 1].curruns--;

		arena->stats.nmalloc_large++;
		arena->stats.nrequests_large++;
		arena->stats.allocated_large += size;
		arena->stats.lstats[(size >> LG_PAGE) - 1].nmalloc++;
		arena->stats.lstats[(size >> LG_PAGE) - 1].nrequests++;
		arena->stats.lstats[(size >> LG_PAGE) - 1].curruns++;
	}
	malloc_mutex_unlock(&arena->lock);
}

static bool
arena_ralloc_large_grow(arena_t *arena, arena_chunk_t *chunk, void *ptr,
    size_t oldsize, size_t size, size_t extra, bool zero)
{
	size_t pageind = ((uintptr_t)ptr - (uintptr_t)chunk) >> LG_PAGE;
	size_t npages = oldsize >> LG_PAGE;
	size_t followsize;

	assert(oldsize == arena_mapbits_large_size_get(chunk, pageind));

	/* Try to extend the run. */
	assert(size + extra > oldsize);
	malloc_mutex_lock(&arena->lock);
	if (pageind + npages < chunk_npages &&
	    arena_mapbits_allocated_get(chunk, pageind+npages) == 0 &&
	    (followsize = arena_mapbits_unallocated_size_get(chunk,
	    pageind+npages)) >= size - oldsize) {
		/*
		 * The next run is available and sufficiently large.  Split the
		 * following run, then merge the first part with the existing
		 * allocation.
		 */
		size_t flag_dirty;
		size_t splitsize = (oldsize + followsize <= size + extra)
		    ? followsize : size + extra - oldsize;
		arena_run_split(arena, (arena_run_t *)((uintptr_t)chunk +
		    ((pageind+npages) << LG_PAGE)), splitsize, true,
		    BININD_INVALID, zero);

		size = oldsize + splitsize;
		npages = size >> LG_PAGE;

		/*
		 * Mark the extended run as dirty if either portion of the run
		 * was dirty before allocation.  This is rather pedantic,
		 * because there's not actually any sequence of events that
		 * could cause the resulting run to be passed to
		 * arena_run_dalloc() with the dirty argument set to false
		 * (which is when dirty flag consistency would really matter).
		 */
		flag_dirty = arena_mapbits_dirty_get(chunk, pageind) |
		    arena_mapbits_dirty_get(chunk, pageind+npages-1);
		arena_mapbits_large_set(chunk, pageind, size, flag_dirty);
		arena_mapbits_large_set(chunk, pageind+npages-1, 0, flag_dirty);

		if (config_stats) {
			arena->stats.ndalloc_large++;
			arena->stats.allocated_large -= oldsize;
			arena->stats.lstats[(oldsize >> LG_PAGE) - 1].ndalloc++;
			arena->stats.lstats[(oldsize >> LG_PAGE) - 1].curruns--;

			arena->stats.nmalloc_large++;
			arena->stats.nrequests_large++;
			arena->stats.allocated_large += size;
			arena->stats.lstats[(size >> LG_PAGE) - 1].nmalloc++;
			arena->stats.lstats[(size >> LG_PAGE) - 1].nrequests++;
			arena->stats.lstats[(size >> LG_PAGE) - 1].curruns++;
		}
		malloc_mutex_unlock(&arena->lock);
		return (false);
	}
	malloc_mutex_unlock(&arena->lock);

	return (true);
}

/*
 * Try to resize a large allocation, in order to avoid copying.  This will
 * always fail if growing an object, and the following run is already in use.
 */
static bool
arena_ralloc_large(void *ptr, size_t oldsize, size_t size, size_t extra,
    bool zero)
{
	size_t psize;

	psize = PAGE_CEILING(size + extra);
	if (psize == oldsize) {
		/* Same size class. */
		if (config_fill && opt_junk && size < oldsize) {
			memset((void *)((uintptr_t)ptr + size), 0x5a, oldsize -
			    size);
		}
		return (false);
	} else {
		arena_chunk_t *chunk;
		arena_t *arena;

		chunk = (arena_chunk_t *)CHUNK_ADDR2BASE(ptr);
		arena = chunk->arena;

		if (psize < oldsize) {
			/* Fill before shrinking in order avoid a race. */
			if (config_fill && opt_junk) {
				memset((void *)((uintptr_t)ptr + size), 0x5a,
				    oldsize - size);
			}
			arena_ralloc_large_shrink(arena, chunk, ptr, oldsize,
			    psize);
			return (false);
		} else {
			bool ret = arena_ralloc_large_grow(arena, chunk, ptr,
			    oldsize, PAGE_CEILING(size),
			    psize - PAGE_CEILING(size), zero);
			if (config_fill && ret == false && zero == false &&
			    opt_zero) {
				memset((void *)((uintptr_t)ptr + oldsize), 0,
				    size - oldsize);
			}
			return (ret);
		}
	}
}

void *
arena_ralloc_no_move(void *ptr, size_t oldsize, size_t size, size_t extra,
    bool zero)
{

	/*
	 * Avoid moving the allocation if the size class can be left the same.
	 */
	if (oldsize <= arena_maxclass) {
		if (oldsize <= SMALL_MAXCLASS) {
			assert(arena_bin_info[SMALL_SIZE2BIN(oldsize)].reg_size
			    == oldsize);
			if ((size + extra <= SMALL_MAXCLASS &&
			    SMALL_SIZE2BIN(size + extra) ==
			    SMALL_SIZE2BIN(oldsize)) || (size <= oldsize &&
			    size + extra >= oldsize)) {
				if (config_fill && opt_junk && size < oldsize) {
					memset((void *)((uintptr_t)ptr + size),
					    0x5a, oldsize - size);
				}
				return (ptr);
			}
		} else {
			assert(size <= arena_maxclass);
			if (size + extra > SMALL_MAXCLASS) {
				if (arena_ralloc_large(ptr, oldsize, size,
				    extra, zero) == false)
					return (ptr);
			}
		}
	}

	/* Reallocation would require a move. */
	return (NULL);
}

void *
arena_ralloc(arena_t *arena, void *ptr, size_t oldsize, size_t size,
    size_t extra, size_t alignment, bool zero, bool try_tcache_alloc,
    bool try_tcache_dalloc)
{
	void *ret;
	size_t copysize;

	/* Try to avoid moving the allocation. */
	ret = arena_ralloc_no_move(ptr, oldsize, size, extra, zero);
	if (ret != NULL)
		return (ret);

	/*
	 * size and oldsize are different enough that we need to move the
	 * object.  In that case, fall back to allocating new space and
	 * copying.
	 */
	if (alignment != 0) {
		size_t usize = sa2u(size + extra, alignment);
		if (usize == 0)
			return (NULL);
		ret = ipallocx(usize, alignment, zero, try_tcache_alloc, arena);
	} else
		ret = arena_malloc(arena, size + extra, zero, try_tcache_alloc);

	if (ret == NULL) {
		if (extra == 0)
			return (NULL);
		/* Try again, this time without extra. */
		if (alignment != 0) {
			size_t usize = sa2u(size, alignment);
			if (usize == 0)
				return (NULL);
			ret = ipallocx(usize, alignment, zero, try_tcache_alloc,
			    arena);
		} else
			ret = arena_malloc(arena, size, zero, try_tcache_alloc);

		if (ret == NULL)
			return (NULL);
	}

	/* Junk/zero-filling were already done by ipalloc()/arena_malloc(). */

	/*
	 * Copy at most size bytes (not size+extra), since the caller has no
	 * expectation that the extra bytes will be reliably preserved.
	 */
	copysize = (size < oldsize) ? size : oldsize;
	VALGRIND_MAKE_MEM_UNDEFINED(ret, copysize);
	memcpy(ret, ptr, copysize);
	iqallocx(ptr, try_tcache_dalloc);
	return (ret);
}

dss_prec_t
arena_dss_prec_get(arena_t *arena)
{
	dss_prec_t ret;

	malloc_mutex_lock(&arena->lock);
	ret = arena->dss_prec;
	malloc_mutex_unlock(&arena->lock);
	return (ret);
}

void
arena_dss_prec_set(arena_t *arena, dss_prec_t dss_prec)
{

	malloc_mutex_lock(&arena->lock);
	arena->dss_prec = dss_prec;
	malloc_mutex_unlock(&arena->lock);
}

void
arena_stats_merge(arena_t *arena, const char **dss, size_t *nactive,
    size_t *ndirty, arena_stats_t *astats, malloc_bin_stats_t *bstats,
    malloc_large_stats_t *lstats)
{
	unsigned i;

	malloc_mutex_lock(&arena->lock);
	*dss = dss_prec_names[arena->dss_prec];
	*nactive += arena->nactive;
	*ndirty += arena->ndirty;

	astats->mapped += arena->stats.mapped;
	astats->npurge += arena->stats.npurge;
	astats->nmadvise += arena->stats.nmadvise;
	astats->purged += arena->stats.purged;
	astats->allocated_large += arena->stats.allocated_large;
	astats->nmalloc_large += arena->stats.nmalloc_large;
	astats->ndalloc_large += arena->stats.ndalloc_large;
	astats->nrequests_large += arena->stats.nrequests_large;

	for (i = 0; i < nlclasses; i++) {
		lstats[i].nmalloc += arena->stats.lstats[i].nmalloc;
		lstats[i].ndalloc += arena->stats.lstats[i].ndalloc;
		lstats[i].nrequests += arena->stats.lstats[i].nrequests;
		lstats[i].curruns += arena->stats.lstats[i].curruns;
	}
	malloc_mutex_unlock(&arena->lock);

	for (i = 0; i < NBINS; i++) {
		arena_bin_t *bin = &arena->bins[i];

		malloc_mutex_lock(&bin->lock);
		bstats[i].allocated += bin->stats.allocated;
		bstats[i].nmalloc += bin->stats.nmalloc;
		bstats[i].ndalloc += bin->stats.ndalloc;
		bstats[i].nrequests += bin->stats.nrequests;
		if (config_tcache) {
			bstats[i].nfills += bin->stats.nfills;
			bstats[i].nflushes += bin->stats.nflushes;
		}
		bstats[i].nruns += bin->stats.nruns;
		bstats[i].reruns += bin->stats.reruns;
		bstats[i].curruns += bin->stats.curruns;
		malloc_mutex_unlock(&bin->lock);
	}
}

bool
arena_new(arena_t *arena, unsigned ind)
{
	unsigned i;
	arena_bin_t *bin;

	arena->ind = ind;
	arena->nthreads = 0;

	if (malloc_mutex_init(&arena->lock))
		return (true);

	if (config_stats) {
		memset(&arena->stats, 0, sizeof(arena_stats_t));
		arena->stats.lstats =
		    (malloc_large_stats_t *)base_alloc(nlclasses *
		    sizeof(malloc_large_stats_t));
		if (arena->stats.lstats == NULL)
			return (true);
		memset(arena->stats.lstats, 0, nlclasses *
		    sizeof(malloc_large_stats_t));
		if (config_tcache)
			ql_new(&arena->tcache_ql);
	}

	if (config_prof)
		arena->prof_accumbytes = 0;

	arena->dss_prec = chunk_dss_prec_get();

	/* Initialize chunks. */
	arena_chunk_dirty_new(&arena->chunks_dirty);
	arena->spare = NULL;

	arena->nactive = 0;
	arena->ndirty = 0;
	arena->npurgatory = 0;

	arena_avail_tree_new(&arena->runs_avail);

	/* Initialize bins. */
	for (i = 0; i < NBINS; i++) {
		bin = &arena->bins[i];
		if (malloc_mutex_init(&bin->lock))
			return (true);
		bin->runcur = NULL;
		arena_run_tree_new(&bin->runs);
		if (config_stats)
			memset(&bin->stats, 0, sizeof(malloc_bin_stats_t));
	}

	return (false);
}

/*
 * Calculate bin_info->run_size such that it meets the following constraints:
 *
 *   *) bin_info->run_size >= min_run_size
 *   *) bin_info->run_size <= arena_maxclass
 *   *) run header overhead <= RUN_MAX_OVRHD (or header overhead relaxed).
 *   *) bin_info->nregs <= RUN_MAXREGS
 *
 * bin_info->nregs, bin_info->bitmap_offset, and bin_info->reg0_offset are also
 * calculated here, since these settings are all interdependent.
 */
static size_t
bin_info_run_size_calc(arena_bin_info_t *bin_info, size_t min_run_size)
{
	size_t pad_size;
	size_t try_run_size, good_run_size;
	uint32_t try_nregs, good_nregs;
	uint32_t try_hdr_size, good_hdr_size;
	uint32_t try_bitmap_offset, good_bitmap_offset;
	uint32_t try_ctx0_offset, good_ctx0_offset;
	uint32_t try_redzone0_offset, good_redzone0_offset;

	assert(min_run_size >= PAGE);
	assert(min_run_size <= arena_maxclass);

	/*
	 * Determine redzone size based on minimum alignment and minimum
	 * redzone size.  Add padding to the end of the run if it is needed to
	 * align the regions.  The padding allows each redzone to be half the
	 * minimum alignment; without the padding, each redzone would have to
	 * be twice as large in order to maintain alignment.
	 */
	if (config_fill && opt_redzone) {
		size_t align_min = ZU(1) << (ffs(bin_info->reg_size) - 1);
		if (align_min <= REDZONE_MINSIZE) {
			bin_info->redzone_size = REDZONE_MINSIZE;
			pad_size = 0;
		} else {
			bin_info->redzone_size = align_min >> 1;
			pad_size = bin_info->redzone_size;
		}
	} else {
		bin_info->redzone_size = 0;
		pad_size = 0;
	}
	bin_info->reg_interval = bin_info->reg_size +
	    (bin_info->redzone_size << 1);

	/*
	 * Calculate known-valid settings before entering the run_size
	 * expansion loop, so that the first part of the loop always copies
	 * valid settings.
	 *
	 * The do..while loop iteratively reduces the number of regions until
	 * the run header and the regions no longer overlap.  A closed formula
	 * would be quite messy, since there is an interdependency between the
	 * header's mask length and the number of regions.
	 */
	try_run_size = min_run_size;
	try_nregs = ((try_run_size - sizeof(arena_run_t)) /
	    bin_info->reg_interval)
	    + 1; /* Counter-act try_nregs-- in loop. */
	if (try_nregs > RUN_MAXREGS) {
		try_nregs = RUN_MAXREGS
		    + 1; /* Counter-act try_nregs-- in loop. */
	}
	do {
		try_nregs--;
		try_hdr_size = sizeof(arena_run_t);
		/* Pad to a long boundary. */
		try_hdr_size = LONG_CEILING(try_hdr_size);
		try_bitmap_offset = try_hdr_size;
		/* Add space for bitmap. */
		try_hdr_size += bitmap_size(try_nregs);
		if (config_prof && opt_prof && prof_promote == false) {
			/* Pad to a quantum boundary. */
			try_hdr_size = QUANTUM_CEILING(try_hdr_size);
			try_ctx0_offset = try_hdr_size;
			/* Add space for one (prof_ctx_t *) per region. */
			try_hdr_size += try_nregs * sizeof(prof_ctx_t *);
		} else
			try_ctx0_offset = 0;
		try_redzone0_offset = try_run_size - (try_nregs *
		    bin_info->reg_interval) - pad_size;
	} while (try_hdr_size > try_redzone0_offset);

	/* run_size expansion loop. */
	do {
		/*
		 * Copy valid settings before trying more aggressive settings.
		 */
		good_run_size = try_run_size;
		good_nregs = try_nregs;
		good_hdr_size = try_hdr_size;
		good_bitmap_offset = try_bitmap_offset;
		good_ctx0_offset = try_ctx0_offset;
		good_redzone0_offset = try_redzone0_offset;

		/* Try more aggressive settings. */
		try_run_size += PAGE;
		try_nregs = ((try_run_size - sizeof(arena_run_t) - pad_size) /
		    bin_info->reg_interval)
		    + 1; /* Counter-act try_nregs-- in loop. */
		if (try_nregs > RUN_MAXREGS) {
			try_nregs = RUN_MAXREGS
			    + 1; /* Counter-act try_nregs-- in loop. */
		}
		do {
			try_nregs--;
			try_hdr_size = sizeof(arena_run_t);
			/* Pad to a long boundary. */
			try_hdr_size = LONG_CEILING(try_hdr_size);
			try_bitmap_offset = try_hdr_size;
			/* Add space for bitmap. */
			try_hdr_size += bitmap_size(try_nregs);
			if (config_prof && opt_prof && prof_promote == false) {
				/* Pad to a quantum boundary. */
				try_hdr_size = QUANTUM_CEILING(try_hdr_size);
				try_ctx0_offset = try_hdr_size;
				/*
				 * Add space for one (prof_ctx_t *) per region.
				 */
				try_hdr_size += try_nregs *
				    sizeof(prof_ctx_t *);
			}
			try_redzone0_offset = try_run_size - (try_nregs *
			    bin_info->reg_interval) - pad_size;
		} while (try_hdr_size > try_redzone0_offset);
	} while (try_run_size <= arena_maxclass
	    && try_run_size <= arena_maxclass
	    && RUN_MAX_OVRHD * (bin_info->reg_interval << 3) >
	    RUN_MAX_OVRHD_RELAX
	    && (try_redzone0_offset << RUN_BFP) > RUN_MAX_OVRHD * try_run_size
	    && try_nregs < RUN_MAXREGS);

	assert(good_hdr_size <= good_redzone0_offset);

	/* Copy final settings. */
	bin_info->run_size = good_run_size;
	bin_info->nregs = good_nregs;
	bin_info->bitmap_offset = good_bitmap_offset;
	bin_info->ctx0_offset = good_ctx0_offset;
	bin_info->reg0_offset = good_redzone0_offset + bin_info->redzone_size;

	assert(bin_info->reg0_offset - bin_info->redzone_size + (bin_info->nregs
	    * bin_info->reg_interval) + pad_size == bin_info->run_size);

	return (good_run_size);
}

static void
bin_info_init(void)
{
	arena_bin_info_t *bin_info;
	size_t prev_run_size = PAGE;

#define	SIZE_CLASS(bin, delta, size)					\
	bin_info = &arena_bin_info[bin];				\
	bin_info->reg_size = size;					\
	prev_run_size = bin_info_run_size_calc(bin_info, prev_run_size);\
	bitmap_info_init(&bin_info->bitmap_info, bin_info->nregs);
	SIZE_CLASSES
#undef SIZE_CLASS
}

void
arena_boot(void)
{
	size_t header_size;
	unsigned i;

	/*
	 * Compute the header size such that it is large enough to contain the
	 * page map.  The page map is biased to omit entries for the header
	 * itself, so some iteration is necessary to compute the map bias.
	 *
	 * 1) Compute safe header_size and map_bias values that include enough
	 *    space for an unbiased page map.
	 * 2) Refine map_bias based on (1) to omit the header pages in the page
	 *    map.  The resulting map_bias may be one too small.
	 * 3) Refine map_bias based on (2).  The result will be >= the result
	 *    from (2), and will always be correct.
	 */
	map_bias = 0;
	for (i = 0; i < 3; i++) {
		header_size = offsetof(arena_chunk_t, map) +
		    (sizeof(arena_chunk_map_t) * (chunk_npages-map_bias));
		map_bias = (header_size >> LG_PAGE) + ((header_size & PAGE_MASK)
		    != 0);
	}
	assert(map_bias > 0);

	arena_maxclass = chunksize - (map_bias << LG_PAGE);

	bin_info_init();
}

void
arena_prefork(arena_t *arena)
{
	unsigned i;

	malloc_mutex_prefork(&arena->lock);
	for (i = 0; i < NBINS; i++)
		malloc_mutex_prefork(&arena->bins[i].lock);
}

void
arena_postfork_parent(arena_t *arena)
{
	unsigned i;

	for (i = 0; i < NBINS; i++)
		malloc_mutex_postfork_parent(&arena->bins[i].lock);
	malloc_mutex_postfork_parent(&arena->lock);
}

void
arena_postfork_child(arena_t *arena)
{
	unsigned i;

	for (i = 0; i < NBINS; i++)
		malloc_mutex_postfork_child(&arena->bins[i].lock);
	malloc_mutex_postfork_child(&arena->lock);
}
