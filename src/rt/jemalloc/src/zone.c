#include "jemalloc/internal/jemalloc_internal.h"
#ifndef JEMALLOC_ZONE
#  error "This source file is for zones on Darwin (OS X)."
#endif

/*
 * The malloc_default_purgeable_zone function is only available on >= 10.6.
 * We need to check whether it is present at runtime, thus the weak_import.
 */
extern malloc_zone_t *malloc_default_purgeable_zone(void)
JEMALLOC_ATTR(weak_import);

/******************************************************************************/
/* Data. */

static malloc_zone_t zone;
static struct malloc_introspection_t zone_introspect;

/******************************************************************************/
/* Function prototypes for non-inline static functions. */

static size_t	zone_size(malloc_zone_t *zone, void *ptr);
static void	*zone_malloc(malloc_zone_t *zone, size_t size);
static void	*zone_calloc(malloc_zone_t *zone, size_t num, size_t size);
static void	*zone_valloc(malloc_zone_t *zone, size_t size);
static void	zone_free(malloc_zone_t *zone, void *ptr);
static void	*zone_realloc(malloc_zone_t *zone, void *ptr, size_t size);
#if (JEMALLOC_ZONE_VERSION >= 5)
static void	*zone_memalign(malloc_zone_t *zone, size_t alignment,
#endif
#if (JEMALLOC_ZONE_VERSION >= 6)
    size_t size);
static void	zone_free_definite_size(malloc_zone_t *zone, void *ptr,
    size_t size);
#endif
static void	*zone_destroy(malloc_zone_t *zone);
static size_t	zone_good_size(malloc_zone_t *zone, size_t size);
static void	zone_force_lock(malloc_zone_t *zone);
static void	zone_force_unlock(malloc_zone_t *zone);

/******************************************************************************/
/*
 * Functions.
 */

static size_t
zone_size(malloc_zone_t *zone, void *ptr)
{

	/*
	 * There appear to be places within Darwin (such as setenv(3)) that
	 * cause calls to this function with pointers that *no* zone owns.  If
	 * we knew that all pointers were owned by *some* zone, we could split
	 * our zone into two parts, and use one as the default allocator and
	 * the other as the default deallocator/reallocator.  Since that will
	 * not work in practice, we must check all pointers to assure that they
	 * reside within a mapped chunk before determining size.
	 */
	return (ivsalloc(ptr, config_prof));
}

static void *
zone_malloc(malloc_zone_t *zone, size_t size)
{

	return (je_malloc(size));
}

static void *
zone_calloc(malloc_zone_t *zone, size_t num, size_t size)
{

	return (je_calloc(num, size));
}

static void *
zone_valloc(malloc_zone_t *zone, size_t size)
{
	void *ret = NULL; /* Assignment avoids useless compiler warning. */

	je_posix_memalign(&ret, PAGE, size);

	return (ret);
}

static void
zone_free(malloc_zone_t *zone, void *ptr)
{

	if (ivsalloc(ptr, config_prof) != 0) {
		je_free(ptr);
		return;
	}

	free(ptr);
}

static void *
zone_realloc(malloc_zone_t *zone, void *ptr, size_t size)
{

	if (ivsalloc(ptr, config_prof) != 0)
		return (je_realloc(ptr, size));

	return (realloc(ptr, size));
}

#if (JEMALLOC_ZONE_VERSION >= 5)
static void *
zone_memalign(malloc_zone_t *zone, size_t alignment, size_t size)
{
	void *ret = NULL; /* Assignment avoids useless compiler warning. */

	je_posix_memalign(&ret, alignment, size);

	return (ret);
}
#endif

#if (JEMALLOC_ZONE_VERSION >= 6)
static void
zone_free_definite_size(malloc_zone_t *zone, void *ptr, size_t size)
{

	if (ivsalloc(ptr, config_prof) != 0) {
		assert(ivsalloc(ptr, config_prof) == size);
		je_free(ptr);
		return;
	}

	free(ptr);
}
#endif

static void *
zone_destroy(malloc_zone_t *zone)
{

	/* This function should never be called. */
	assert(false);
	return (NULL);
}

static size_t
zone_good_size(malloc_zone_t *zone, size_t size)
{

	if (size == 0)
		size = 1;
	return (s2u(size));
}

static void
zone_force_lock(malloc_zone_t *zone)
{

	if (isthreaded)
		jemalloc_prefork();
}

static void
zone_force_unlock(malloc_zone_t *zone)
{

	if (isthreaded)
		jemalloc_postfork_parent();
}

JEMALLOC_ATTR(constructor)
void
register_zone(void)
{

	/*
	 * If something else replaced the system default zone allocator, don't
	 * register jemalloc's.
	 */
	malloc_zone_t *default_zone = malloc_default_zone();
	if (!default_zone->zone_name ||
	    strcmp(default_zone->zone_name, "DefaultMallocZone") != 0) {
		return;
	}

	zone.size = (void *)zone_size;
	zone.malloc = (void *)zone_malloc;
	zone.calloc = (void *)zone_calloc;
	zone.valloc = (void *)zone_valloc;
	zone.free = (void *)zone_free;
	zone.realloc = (void *)zone_realloc;
	zone.destroy = (void *)zone_destroy;
	zone.zone_name = "jemalloc_zone";
	zone.batch_malloc = NULL;
	zone.batch_free = NULL;
	zone.introspect = &zone_introspect;
	zone.version = JEMALLOC_ZONE_VERSION;
#if (JEMALLOC_ZONE_VERSION >= 5)
	zone.memalign = zone_memalign;
#endif
#if (JEMALLOC_ZONE_VERSION >= 6)
	zone.free_definite_size = zone_free_definite_size;
#endif
#if (JEMALLOC_ZONE_VERSION >= 8)
	zone.pressure_relief = NULL;
#endif

	zone_introspect.enumerator = NULL;
	zone_introspect.good_size = (void *)zone_good_size;
	zone_introspect.check = NULL;
	zone_introspect.print = NULL;
	zone_introspect.log = NULL;
	zone_introspect.force_lock = (void *)zone_force_lock;
	zone_introspect.force_unlock = (void *)zone_force_unlock;
	zone_introspect.statistics = NULL;
#if (JEMALLOC_ZONE_VERSION >= 6)
	zone_introspect.zone_locked = NULL;
#endif
#if (JEMALLOC_ZONE_VERSION >= 7)
	zone_introspect.enable_discharge_checking = NULL;
	zone_introspect.disable_discharge_checking = NULL;
	zone_introspect.discharge = NULL;
#ifdef __BLOCKS__
	zone_introspect.enumerate_discharged_pointers = NULL;
#else
	zone_introspect.enumerate_unavailable_without_blocks = NULL;
#endif
#endif

	/*
	 * The default purgeable zone is created lazily by OSX's libc.  It uses
	 * the default zone when it is created for "small" allocations
	 * (< 15 KiB), but assumes the default zone is a scalable_zone.  This
	 * obviously fails when the default zone is the jemalloc zone, so
	 * malloc_default_purgeable_zone is called beforehand so that the
	 * default purgeable zone is created when the default zone is still
	 * a scalable_zone.  As purgeable zones only exist on >= 10.6, we need
	 * to check for the existence of malloc_default_purgeable_zone() at
	 * run time.
	 */
	if (malloc_default_purgeable_zone != NULL)
		malloc_default_purgeable_zone();

	/* Register the custom zone.  At this point it won't be the default. */
	malloc_zone_register(&zone);

	/*
	 * Unregister and reregister the default zone.  On OSX >= 10.6,
	 * unregistering takes the last registered zone and places it at the
	 * location of the specified zone.  Unregistering the default zone thus
	 * makes the last registered one the default.  On OSX < 10.6,
	 * unregistering shifts all registered zones.  The first registered zone
	 * then becomes the default.
	 */
	do {
		default_zone = malloc_default_zone();
		malloc_zone_unregister(default_zone);
		malloc_zone_register(default_zone);
	} while (malloc_default_zone() != &zone);
}
