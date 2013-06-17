#define	JEMALLOC_MANGLE
#include "jemalloc_test.h"

int
main(void)
{
	size_t pagesize;
	void *p, *q;
	size_t sz, tsz;
	int r;

	malloc_printf("Test begin\n");

	/* Get page size. */
	{
#ifdef _WIN32
		SYSTEM_INFO si;
		GetSystemInfo(&si);
		pagesize = (size_t)si.dwPageSize;
#else
		long result = sysconf(_SC_PAGESIZE);
		assert(result != -1);
		pagesize = (size_t)result;
#endif
	}

	r = allocm(&p, &sz, 42, 0);
	if (r != ALLOCM_SUCCESS) {
		malloc_printf("Unexpected allocm() error\n");
		abort();
	}

	q = p;
	r = rallocm(&q, &tsz, sz, 0, ALLOCM_NO_MOVE);
	if (r != ALLOCM_SUCCESS)
		malloc_printf("Unexpected rallocm() error\n");
	if (q != p)
		malloc_printf("Unexpected object move\n");
	if (tsz != sz) {
		malloc_printf("Unexpected size change: %zu --> %zu\n",
		    sz, tsz);
	}

	q = p;
	r = rallocm(&q, &tsz, sz, 5, ALLOCM_NO_MOVE);
	if (r != ALLOCM_SUCCESS)
		malloc_printf("Unexpected rallocm() error\n");
	if (q != p)
		malloc_printf("Unexpected object move\n");
	if (tsz != sz) {
		malloc_printf("Unexpected size change: %zu --> %zu\n",
		    sz, tsz);
	}

	q = p;
	r = rallocm(&q, &tsz, sz + 5, 0, ALLOCM_NO_MOVE);
	if (r != ALLOCM_ERR_NOT_MOVED)
		malloc_printf("Unexpected rallocm() result\n");
	if (q != p)
		malloc_printf("Unexpected object move\n");
	if (tsz != sz) {
		malloc_printf("Unexpected size change: %zu --> %zu\n",
		    sz, tsz);
	}

	q = p;
	r = rallocm(&q, &tsz, sz + 5, 0, 0);
	if (r != ALLOCM_SUCCESS)
		malloc_printf("Unexpected rallocm() error\n");
	if (q == p)
		malloc_printf("Expected object move\n");
	if (tsz == sz) {
		malloc_printf("Expected size change: %zu --> %zu\n",
		    sz, tsz);
	}
	p = q;
	sz = tsz;

	r = rallocm(&q, &tsz, pagesize*2, 0, 0);
	if (r != ALLOCM_SUCCESS)
		malloc_printf("Unexpected rallocm() error\n");
	if (q == p)
		malloc_printf("Expected object move\n");
	if (tsz == sz) {
		malloc_printf("Expected size change: %zu --> %zu\n",
		    sz, tsz);
	}
	p = q;
	sz = tsz;

	r = rallocm(&q, &tsz, pagesize*4, 0, 0);
	if (r != ALLOCM_SUCCESS)
		malloc_printf("Unexpected rallocm() error\n");
	if (tsz == sz) {
		malloc_printf("Expected size change: %zu --> %zu\n",
		    sz, tsz);
	}
	p = q;
	sz = tsz;

	r = rallocm(&q, &tsz, pagesize*2, 0, ALLOCM_NO_MOVE);
	if (r != ALLOCM_SUCCESS)
		malloc_printf("Unexpected rallocm() error\n");
	if (q != p)
		malloc_printf("Unexpected object move\n");
	if (tsz == sz) {
		malloc_printf("Expected size change: %zu --> %zu\n",
		    sz, tsz);
	}
	sz = tsz;

	r = rallocm(&q, &tsz, pagesize*4, 0, ALLOCM_NO_MOVE);
	if (r != ALLOCM_SUCCESS)
		malloc_printf("Unexpected rallocm() error\n");
	if (q != p)
		malloc_printf("Unexpected object move\n");
	if (tsz == sz) {
		malloc_printf("Expected size change: %zu --> %zu\n",
		    sz, tsz);
	}
	sz = tsz;

	dallocm(p, 0);

	malloc_printf("Test end\n");
	return (0);
}
