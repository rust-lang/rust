#define	JEMALLOC_HUGE_C_
#include "jemalloc/internal/jemalloc_internal.h"

/******************************************************************************/
/* Data. */

uint64_t	huge_nmalloc;
uint64_t	huge_ndalloc;
size_t		huge_allocated;

malloc_mutex_t	huge_mtx;

/******************************************************************************/

/* Tree of chunks that are stand-alone huge allocations. */
static extent_tree_t	huge;

void *
huge_malloc(size_t size, bool zero)
{

	return (huge_palloc(size, chunksize, zero));
}

void *
huge_palloc(size_t size, size_t alignment, bool zero)
{
	void *ret;
	size_t csize;
	extent_node_t *node;
	bool is_zeroed;

	/* Allocate one or more contiguous chunks for this request. */

	csize = CHUNK_CEILING(size);
	if (csize == 0) {
		/* size is large enough to cause size_t wrap-around. */
		return (NULL);
	}

	/* Allocate an extent node with which to track the chunk. */
	node = base_node_alloc();
	if (node == NULL)
		return (NULL);

	/*
	 * Copy zero into is_zeroed and pass the copy to chunk_alloc(), so that
	 * it is possible to make correct junk/zero fill decisions below.
	 */
	is_zeroed = zero;
	ret = chunk_alloc(csize, alignment, false, &is_zeroed,
	    chunk_dss_prec_get());
	if (ret == NULL) {
		base_node_dealloc(node);
		return (NULL);
	}

	/* Insert node into huge. */
	node->addr = ret;
	node->size = csize;

	malloc_mutex_lock(&huge_mtx);
	extent_tree_ad_insert(&huge, node);
	if (config_stats) {
		stats_cactive_add(csize);
		huge_nmalloc++;
		huge_allocated += csize;
	}
	malloc_mutex_unlock(&huge_mtx);

	if (config_fill && zero == false) {
		if (opt_junk)
			memset(ret, 0xa5, csize);
		else if (opt_zero && is_zeroed == false)
			memset(ret, 0, csize);
	}

	return (ret);
}

void *
huge_ralloc_no_move(void *ptr, size_t oldsize, size_t size, size_t extra)
{

	/*
	 * Avoid moving the allocation if the size class can be left the same.
	 */
	if (oldsize > arena_maxclass
	    && CHUNK_CEILING(oldsize) >= CHUNK_CEILING(size)
	    && CHUNK_CEILING(oldsize) <= CHUNK_CEILING(size+extra)) {
		assert(CHUNK_CEILING(oldsize) == oldsize);
		if (config_fill && opt_junk && size < oldsize) {
			memset((void *)((uintptr_t)ptr + size), 0x5a,
			    oldsize - size);
		}
		return (ptr);
	}

	/* Reallocation would require a move. */
	return (NULL);
}

void *
huge_ralloc(void *ptr, size_t oldsize, size_t size, size_t extra,
    size_t alignment, bool zero, bool try_tcache_dalloc)
{
	void *ret;
	size_t copysize;

	/* Try to avoid moving the allocation. */
	ret = huge_ralloc_no_move(ptr, oldsize, size, extra);
	if (ret != NULL)
		return (ret);

	/*
	 * size and oldsize are different enough that we need to use a
	 * different size class.  In that case, fall back to allocating new
	 * space and copying.
	 */
	if (alignment > chunksize)
		ret = huge_palloc(size + extra, alignment, zero);
	else
		ret = huge_malloc(size + extra, zero);

	if (ret == NULL) {
		if (extra == 0)
			return (NULL);
		/* Try again, this time without extra. */
		if (alignment > chunksize)
			ret = huge_palloc(size, alignment, zero);
		else
			ret = huge_malloc(size, zero);

		if (ret == NULL)
			return (NULL);
	}

	/*
	 * Copy at most size bytes (not size+extra), since the caller has no
	 * expectation that the extra bytes will be reliably preserved.
	 */
	copysize = (size < oldsize) ? size : oldsize;

#ifdef JEMALLOC_MREMAP
	/*
	 * Use mremap(2) if this is a huge-->huge reallocation, and neither the
	 * source nor the destination are in dss.
	 */
	if (oldsize >= chunksize && (config_dss == false || (chunk_in_dss(ptr)
	    == false && chunk_in_dss(ret) == false))) {
		size_t newsize = huge_salloc(ret);

		/*
		 * Remove ptr from the tree of huge allocations before
		 * performing the remap operation, in order to avoid the
		 * possibility of another thread acquiring that mapping before
		 * this one removes it from the tree.
		 */
		huge_dalloc(ptr, false);
		if (mremap(ptr, oldsize, newsize, MREMAP_MAYMOVE|MREMAP_FIXED,
		    ret) == MAP_FAILED) {
			/*
			 * Assuming no chunk management bugs in the allocator,
			 * the only documented way an error can occur here is
			 * if the application changed the map type for a
			 * portion of the old allocation.  This is firmly in
			 * undefined behavior territory, so write a diagnostic
			 * message, and optionally abort.
			 */
			char buf[BUFERROR_BUF];

			buferror(buf, sizeof(buf));
			malloc_printf("<jemalloc>: Error in mremap(): %s\n",
			    buf);
			if (opt_abort)
				abort();
			memcpy(ret, ptr, copysize);
			chunk_dealloc_mmap(ptr, oldsize);
		}
	} else
#endif
	{
		memcpy(ret, ptr, copysize);
		iqallocx(ptr, try_tcache_dalloc);
	}
	return (ret);
}

void
huge_dalloc(void *ptr, bool unmap)
{
	extent_node_t *node, key;

	malloc_mutex_lock(&huge_mtx);

	/* Extract from tree of huge allocations. */
	key.addr = ptr;
	node = extent_tree_ad_search(&huge, &key);
	assert(node != NULL);
	assert(node->addr == ptr);
	extent_tree_ad_remove(&huge, node);

	if (config_stats) {
		stats_cactive_sub(node->size);
		huge_ndalloc++;
		huge_allocated -= node->size;
	}

	malloc_mutex_unlock(&huge_mtx);

	if (unmap && config_fill && config_dss && opt_junk)
		memset(node->addr, 0x5a, node->size);

	chunk_dealloc(node->addr, node->size, unmap);

	base_node_dealloc(node);
}

size_t
huge_salloc(const void *ptr)
{
	size_t ret;
	extent_node_t *node, key;

	malloc_mutex_lock(&huge_mtx);

	/* Extract from tree of huge allocations. */
	key.addr = __DECONST(void *, ptr);
	node = extent_tree_ad_search(&huge, &key);
	assert(node != NULL);

	ret = node->size;

	malloc_mutex_unlock(&huge_mtx);

	return (ret);
}

prof_ctx_t *
huge_prof_ctx_get(const void *ptr)
{
	prof_ctx_t *ret;
	extent_node_t *node, key;

	malloc_mutex_lock(&huge_mtx);

	/* Extract from tree of huge allocations. */
	key.addr = __DECONST(void *, ptr);
	node = extent_tree_ad_search(&huge, &key);
	assert(node != NULL);

	ret = node->prof_ctx;

	malloc_mutex_unlock(&huge_mtx);

	return (ret);
}

void
huge_prof_ctx_set(const void *ptr, prof_ctx_t *ctx)
{
	extent_node_t *node, key;

	malloc_mutex_lock(&huge_mtx);

	/* Extract from tree of huge allocations. */
	key.addr = __DECONST(void *, ptr);
	node = extent_tree_ad_search(&huge, &key);
	assert(node != NULL);

	node->prof_ctx = ctx;

	malloc_mutex_unlock(&huge_mtx);
}

bool
huge_boot(void)
{

	/* Initialize chunks data. */
	if (malloc_mutex_init(&huge_mtx))
		return (true);
	extent_tree_ad_new(&huge);

	if (config_stats) {
		huge_nmalloc = 0;
		huge_ndalloc = 0;
		huge_allocated = 0;
	}

	return (false);
}

void
huge_prefork(void)
{

	malloc_mutex_prefork(&huge_mtx);
}

void
huge_postfork_parent(void)
{

	malloc_mutex_postfork_parent(&huge_mtx);
}

void
huge_postfork_child(void)
{

	malloc_mutex_postfork_child(&huge_mtx);
}
