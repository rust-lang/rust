#define	JEMALLOC_MANGLE
#include "jemalloc_test.h"

void *
je_thread_start(void *arg)
{
	int err;
	void *p;
	uint64_t a0, a1, d0, d1;
	uint64_t *ap0, *ap1, *dp0, *dp1;
	size_t sz, usize;

	sz = sizeof(a0);
	if ((err = mallctl("thread.allocated", &a0, &sz, NULL, 0))) {
		if (err == ENOENT) {
#ifdef JEMALLOC_STATS
			assert(false);
#endif
			goto label_return;
		}
		malloc_printf("%s(): Error in mallctl(): %s\n", __func__,
		    strerror(err));
		exit(1);
	}
	sz = sizeof(ap0);
	if ((err = mallctl("thread.allocatedp", &ap0, &sz, NULL, 0))) {
		if (err == ENOENT) {
#ifdef JEMALLOC_STATS
			assert(false);
#endif
			goto label_return;
		}
		malloc_printf("%s(): Error in mallctl(): %s\n", __func__,
		    strerror(err));
		exit(1);
	}
	assert(*ap0 == a0);

	sz = sizeof(d0);
	if ((err = mallctl("thread.deallocated", &d0, &sz, NULL, 0))) {
		if (err == ENOENT) {
#ifdef JEMALLOC_STATS
			assert(false);
#endif
			goto label_return;
		}
		malloc_printf("%s(): Error in mallctl(): %s\n", __func__,
		    strerror(err));
		exit(1);
	}
	sz = sizeof(dp0);
	if ((err = mallctl("thread.deallocatedp", &dp0, &sz, NULL, 0))) {
		if (err == ENOENT) {
#ifdef JEMALLOC_STATS
			assert(false);
#endif
			goto label_return;
		}
		malloc_printf("%s(): Error in mallctl(): %s\n", __func__,
		    strerror(err));
		exit(1);
	}
	assert(*dp0 == d0);

	p = malloc(1);
	if (p == NULL) {
		malloc_printf("%s(): Error in malloc()\n", __func__);
		exit(1);
	}

	sz = sizeof(a1);
	mallctl("thread.allocated", &a1, &sz, NULL, 0);
	sz = sizeof(ap1);
	mallctl("thread.allocatedp", &ap1, &sz, NULL, 0);
	assert(*ap1 == a1);
	assert(ap0 == ap1);

	usize = malloc_usable_size(p);
	assert(a0 + usize <= a1);

	free(p);

	sz = sizeof(d1);
	mallctl("thread.deallocated", &d1, &sz, NULL, 0);
	sz = sizeof(dp1);
	mallctl("thread.deallocatedp", &dp1, &sz, NULL, 0);
	assert(*dp1 == d1);
	assert(dp0 == dp1);

	assert(d0 + usize <= d1);

label_return:
	return (NULL);
}

int
main(void)
{
	int ret = 0;
	je_thread_t thread;

	malloc_printf("Test begin\n");

	je_thread_start(NULL);

	je_thread_create(&thread, je_thread_start, NULL);
	je_thread_join(thread, (void *)&ret);

	je_thread_start(NULL);

	je_thread_create(&thread, je_thread_start, NULL);
	je_thread_join(thread, (void *)&ret);

	je_thread_start(NULL);

	malloc_printf("Test end\n");
	return (ret);
}
