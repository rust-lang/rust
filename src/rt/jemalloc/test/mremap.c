#define	JEMALLOC_MANGLE
#include "jemalloc_test.h"

int
main(void)
{
	int ret, err;
	size_t sz, lg_chunk, chunksize, i;
	char *p, *q;

	malloc_printf("Test begin\n");

	sz = sizeof(lg_chunk);
	if ((err = mallctl("opt.lg_chunk", &lg_chunk, &sz, NULL, 0))) {
		assert(err != ENOENT);
		malloc_printf("%s(): Error in mallctl(): %s\n", __func__,
		    strerror(err));
		ret = 1;
		goto label_return;
	}
	chunksize = ((size_t)1U) << lg_chunk;

	p = (char *)malloc(chunksize);
	if (p == NULL) {
		malloc_printf("malloc(%zu) --> %p\n", chunksize, p);
		ret = 1;
		goto label_return;
	}
	memset(p, 'a', chunksize);

	q = (char *)realloc(p, chunksize * 2);
	if (q == NULL) {
		malloc_printf("realloc(%p, %zu) --> %p\n", p, chunksize * 2,
		    q);
		ret = 1;
		goto label_return;
	}
	for (i = 0; i < chunksize; i++) {
		assert(q[i] == 'a');
	}

	p = q;

	q = (char *)realloc(p, chunksize);
	if (q == NULL) {
		malloc_printf("realloc(%p, %zu) --> %p\n", p, chunksize, q);
		ret = 1;
		goto label_return;
	}
	for (i = 0; i < chunksize; i++) {
		assert(q[i] == 'a');
	}

	free(q);

	ret = 0;
label_return:
	malloc_printf("Test end\n");
	return (ret);
}
