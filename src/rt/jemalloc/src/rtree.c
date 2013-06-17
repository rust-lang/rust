#define	JEMALLOC_RTREE_C_
#include "jemalloc/internal/jemalloc_internal.h"

rtree_t *
rtree_new(unsigned bits)
{
	rtree_t *ret;
	unsigned bits_per_level, height, i;

	bits_per_level = ffs(pow2_ceil((RTREE_NODESIZE / sizeof(void *)))) - 1;
	height = bits / bits_per_level;
	if (height * bits_per_level != bits)
		height++;
	assert(height * bits_per_level >= bits);

	ret = (rtree_t*)base_alloc(offsetof(rtree_t, level2bits) +
	    (sizeof(unsigned) * height));
	if (ret == NULL)
		return (NULL);
	memset(ret, 0, offsetof(rtree_t, level2bits) + (sizeof(unsigned) *
	    height));

	if (malloc_mutex_init(&ret->mutex)) {
		/* Leak the rtree. */
		return (NULL);
	}
	ret->height = height;
	if (bits_per_level * height > bits)
		ret->level2bits[0] = bits % bits_per_level;
	else
		ret->level2bits[0] = bits_per_level;
	for (i = 1; i < height; i++)
		ret->level2bits[i] = bits_per_level;

	ret->root = (void**)base_alloc(sizeof(void *) << ret->level2bits[0]);
	if (ret->root == NULL) {
		/*
		 * We leak the rtree here, since there's no generic base
		 * deallocation.
		 */
		return (NULL);
	}
	memset(ret->root, 0, sizeof(void *) << ret->level2bits[0]);

	return (ret);
}

void
rtree_prefork(rtree_t *rtree)
{

	malloc_mutex_prefork(&rtree->mutex);
}

void
rtree_postfork_parent(rtree_t *rtree)
{

	malloc_mutex_postfork_parent(&rtree->mutex);
}

void
rtree_postfork_child(rtree_t *rtree)
{

	malloc_mutex_postfork_child(&rtree->mutex);
}
