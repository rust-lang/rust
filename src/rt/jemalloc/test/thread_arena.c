#define	JEMALLOC_MANGLE
#include "jemalloc_test.h"

#define	NTHREADS 10

void *
je_thread_start(void *arg)
{
	unsigned main_arena_ind = *(unsigned *)arg;
	void *p;
	unsigned arena_ind;
	size_t size;
	int err;

	p = malloc(1);
	if (p == NULL) {
		malloc_printf("%s(): Error in malloc()\n", __func__);
		return (void *)1;
	}
	free(p);

	size = sizeof(arena_ind);
	if ((err = mallctl("thread.arena", &arena_ind, &size, &main_arena_ind,
	    sizeof(main_arena_ind)))) {
		malloc_printf("%s(): Error in mallctl(): %s\n", __func__,
		    strerror(err));
		return (void *)1;
	}

	size = sizeof(arena_ind);
	if ((err = mallctl("thread.arena", &arena_ind, &size, NULL,
	    0))) {
		malloc_printf("%s(): Error in mallctl(): %s\n", __func__,
		    strerror(err));
		return (void *)1;
	}
	assert(arena_ind == main_arena_ind);

	return (NULL);
}

int
main(void)
{
	int ret = 0;
	void *p;
	unsigned arena_ind;
	size_t size;
	int err;
	je_thread_t threads[NTHREADS];
	unsigned i;

	malloc_printf("Test begin\n");

	p = malloc(1);
	if (p == NULL) {
		malloc_printf("%s(): Error in malloc()\n", __func__);
		ret = 1;
		goto label_return;
	}

	size = sizeof(arena_ind);
	if ((err = mallctl("thread.arena", &arena_ind, &size, NULL, 0))) {
		malloc_printf("%s(): Error in mallctl(): %s\n", __func__,
		    strerror(err));
		ret = 1;
		goto label_return;
	}

	for (i = 0; i < NTHREADS; i++) {
		je_thread_create(&threads[i], je_thread_start,
		    (void *)&arena_ind);
	}

	for (i = 0; i < NTHREADS; i++)
		je_thread_join(threads[i], (void *)&ret);

label_return:
	malloc_printf("Test end\n");
	return (ret);
}
