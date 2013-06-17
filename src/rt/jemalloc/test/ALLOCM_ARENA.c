#define	JEMALLOC_MANGLE
#include "jemalloc_test.h"

#define	NTHREADS 10

void *
je_thread_start(void *arg)
{
	unsigned thread_ind = (unsigned)(uintptr_t)arg;
	unsigned arena_ind;
	int r;
	void *p;
	size_t rsz, sz;

	sz = sizeof(arena_ind);
	if (mallctl("arenas.extend", &arena_ind, &sz, NULL, 0)
	    != 0) {
		malloc_printf("Error in arenas.extend\n");
		abort();
	}

	if (thread_ind % 4 != 3) {
		size_t mib[3];
		size_t miblen = sizeof(mib) / sizeof(size_t);
		const char *dss_precs[] = {"disabled", "primary", "secondary"};
		const char *dss = dss_precs[thread_ind % 4];
		if (mallctlnametomib("arena.0.dss", mib, &miblen) != 0) {
			malloc_printf("Error in mallctlnametomib()\n");
			abort();
		}
		mib[1] = arena_ind;
		if (mallctlbymib(mib, miblen, NULL, NULL, (void *)&dss,
		    sizeof(const char *))) {
			malloc_printf("Error in mallctlbymib()\n");
			abort();
		}
	}

	r = allocm(&p, &rsz, 1, ALLOCM_ARENA(arena_ind));
	if (r != ALLOCM_SUCCESS) {
		malloc_printf("Unexpected allocm() error\n");
		abort();
	}
	dallocm(p, 0);

	return (NULL);
}

int
main(void)
{
	je_thread_t threads[NTHREADS];
	unsigned i;

	malloc_printf("Test begin\n");

	for (i = 0; i < NTHREADS; i++) {
		je_thread_create(&threads[i], je_thread_start,
		    (void *)(uintptr_t)i);
	}

	for (i = 0; i < NTHREADS; i++)
		je_thread_join(threads[i], NULL);

	malloc_printf("Test end\n");
	return (0);
}
