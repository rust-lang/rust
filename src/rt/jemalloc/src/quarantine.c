#define	JEMALLOC_QUARANTINE_C_
#include "jemalloc/internal/jemalloc_internal.h"

/*
 * quarantine pointers close to NULL are used to encode state information that
 * is used for cleaning up during thread shutdown.
 */
#define	QUARANTINE_STATE_REINCARNATED	((quarantine_t *)(uintptr_t)1)
#define	QUARANTINE_STATE_PURGATORY	((quarantine_t *)(uintptr_t)2)
#define	QUARANTINE_STATE_MAX		QUARANTINE_STATE_PURGATORY

/******************************************************************************/
/* Data. */

malloc_tsd_data(, quarantine, quarantine_t *, NULL)

/******************************************************************************/
/* Function prototypes for non-inline static functions. */

static quarantine_t	*quarantine_grow(quarantine_t *quarantine);
static void	quarantine_drain_one(quarantine_t *quarantine);
static void	quarantine_drain(quarantine_t *quarantine, size_t upper_bound);

/******************************************************************************/

quarantine_t *
quarantine_init(size_t lg_maxobjs)
{
	quarantine_t *quarantine;

	quarantine = (quarantine_t *)imalloc(offsetof(quarantine_t, objs) +
	    ((ZU(1) << lg_maxobjs) * sizeof(quarantine_obj_t)));
	if (quarantine == NULL)
		return (NULL);
	quarantine->curbytes = 0;
	quarantine->curobjs = 0;
	quarantine->first = 0;
	quarantine->lg_maxobjs = lg_maxobjs;

	quarantine_tsd_set(&quarantine);

	return (quarantine);
}

static quarantine_t *
quarantine_grow(quarantine_t *quarantine)
{
	quarantine_t *ret;

	ret = quarantine_init(quarantine->lg_maxobjs + 1);
	if (ret == NULL) {
		quarantine_drain_one(quarantine);
		return (quarantine);
	}

	ret->curbytes = quarantine->curbytes;
	ret->curobjs = quarantine->curobjs;
	if (quarantine->first + quarantine->curobjs <= (ZU(1) <<
	    quarantine->lg_maxobjs)) {
		/* objs ring buffer data are contiguous. */
		memcpy(ret->objs, &quarantine->objs[quarantine->first],
		    quarantine->curobjs * sizeof(quarantine_obj_t));
	} else {
		/* objs ring buffer data wrap around. */
		size_t ncopy_a = (ZU(1) << quarantine->lg_maxobjs) -
		    quarantine->first;
		size_t ncopy_b = quarantine->curobjs - ncopy_a;

		memcpy(ret->objs, &quarantine->objs[quarantine->first], ncopy_a
		    * sizeof(quarantine_obj_t));
		memcpy(&ret->objs[ncopy_a], quarantine->objs, ncopy_b *
		    sizeof(quarantine_obj_t));
	}
	idalloc(quarantine);

	return (ret);
}

static void
quarantine_drain_one(quarantine_t *quarantine)
{
	quarantine_obj_t *obj = &quarantine->objs[quarantine->first];
	assert(obj->usize == isalloc(obj->ptr, config_prof));
	idalloc(obj->ptr);
	quarantine->curbytes -= obj->usize;
	quarantine->curobjs--;
	quarantine->first = (quarantine->first + 1) & ((ZU(1) <<
	    quarantine->lg_maxobjs) - 1);
}

static void
quarantine_drain(quarantine_t *quarantine, size_t upper_bound)
{

	while (quarantine->curbytes > upper_bound && quarantine->curobjs > 0)
		quarantine_drain_one(quarantine);
}

void
quarantine(void *ptr)
{
	quarantine_t *quarantine;
	size_t usize = isalloc(ptr, config_prof);

	cassert(config_fill);
	assert(opt_quarantine);

	quarantine = *quarantine_tsd_get();
	if ((uintptr_t)quarantine <= (uintptr_t)QUARANTINE_STATE_MAX) {
		if (quarantine == QUARANTINE_STATE_PURGATORY) {
			/*
			 * Make a note that quarantine() was called after
			 * quarantine_cleanup() was called.
			 */
			quarantine = QUARANTINE_STATE_REINCARNATED;
			quarantine_tsd_set(&quarantine);
		}
		idalloc(ptr);
		return;
	}
	/*
	 * Drain one or more objects if the quarantine size limit would be
	 * exceeded by appending ptr.
	 */
	if (quarantine->curbytes + usize > opt_quarantine) {
		size_t upper_bound = (opt_quarantine >= usize) ? opt_quarantine
		    - usize : 0;
		quarantine_drain(quarantine, upper_bound);
	}
	/* Grow the quarantine ring buffer if it's full. */
	if (quarantine->curobjs == (ZU(1) << quarantine->lg_maxobjs))
		quarantine = quarantine_grow(quarantine);
	/* quarantine_grow() must free a slot if it fails to grow. */
	assert(quarantine->curobjs < (ZU(1) << quarantine->lg_maxobjs));
	/* Append ptr if its size doesn't exceed the quarantine size. */
	if (quarantine->curbytes + usize <= opt_quarantine) {
		size_t offset = (quarantine->first + quarantine->curobjs) &
		    ((ZU(1) << quarantine->lg_maxobjs) - 1);
		quarantine_obj_t *obj = &quarantine->objs[offset];
		obj->ptr = ptr;
		obj->usize = usize;
		quarantine->curbytes += usize;
		quarantine->curobjs++;
		if (opt_junk)
			memset(ptr, 0x5a, usize);
	} else {
		assert(quarantine->curbytes == 0);
		idalloc(ptr);
	}
}

void
quarantine_cleanup(void *arg)
{
	quarantine_t *quarantine = *(quarantine_t **)arg;

	if (quarantine == QUARANTINE_STATE_REINCARNATED) {
		/*
		 * Another destructor deallocated memory after this destructor
		 * was called.  Reset quarantine to QUARANTINE_STATE_PURGATORY
		 * in order to receive another callback.
		 */
		quarantine = QUARANTINE_STATE_PURGATORY;
		quarantine_tsd_set(&quarantine);
	} else if (quarantine == QUARANTINE_STATE_PURGATORY) {
		/*
		 * The previous time this destructor was called, we set the key
		 * to QUARANTINE_STATE_PURGATORY so that other destructors
		 * wouldn't cause re-creation of the quarantine.  This time, do
		 * nothing, so that the destructor will not be called again.
		 */
	} else if (quarantine != NULL) {
		quarantine_drain(quarantine, 0);
		idalloc(quarantine);
		quarantine = QUARANTINE_STATE_PURGATORY;
		quarantine_tsd_set(&quarantine);
	}
}

bool
quarantine_boot(void)
{

	cassert(config_fill);

	if (quarantine_tsd_boot())
		return (true);

	return (false);
}
