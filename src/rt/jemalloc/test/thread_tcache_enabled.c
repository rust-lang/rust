#define	JEMALLOC_MANGLE
#include "jemalloc_test.h"

void *
je_thread_start(void *arg)
{
	int err;
	size_t sz;
	bool e0, e1;

	sz = sizeof(bool);
	if ((err = mallctl("thread.tcache.enabled", &e0, &sz, NULL, 0))) {
		if (err == ENOENT) {
#ifdef JEMALLOC_TCACHE
			assert(false);
#endif
		}
		goto label_return;
	}

	if (e0) {
		e1 = false;
		assert(mallctl("thread.tcache.enabled", &e0, &sz, &e1, sz)
		    == 0);
		assert(e0);
	}

	e1 = true;
	assert(mallctl("thread.tcache.enabled", &e0, &sz, &e1, sz) == 0);
	assert(e0 == false);

	e1 = true;
	assert(mallctl("thread.tcache.enabled", &e0, &sz, &e1, sz) == 0);
	assert(e0);

	e1 = false;
	assert(mallctl("thread.tcache.enabled", &e0, &sz, &e1, sz) == 0);
	assert(e0);

	e1 = false;
	assert(mallctl("thread.tcache.enabled", &e0, &sz, &e1, sz) == 0);
	assert(e0 == false);

	free(malloc(1));
	e1 = true;
	assert(mallctl("thread.tcache.enabled", &e0, &sz, &e1, sz) == 0);
	assert(e0 == false);

	free(malloc(1));
	e1 = true;
	assert(mallctl("thread.tcache.enabled", &e0, &sz, &e1, sz) == 0);
	assert(e0);

	free(malloc(1));
	e1 = false;
	assert(mallctl("thread.tcache.enabled", &e0, &sz, &e1, sz) == 0);
	assert(e0);

	free(malloc(1));
	e1 = false;
	assert(mallctl("thread.tcache.enabled", &e0, &sz, &e1, sz) == 0);
	assert(e0 == false);

	free(malloc(1));
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
