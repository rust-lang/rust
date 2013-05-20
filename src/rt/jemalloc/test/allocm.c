#define	JEMALLOC_MANGLE
#include "jemalloc_test.h"

#define CHUNK 0x400000
/* #define MAXALIGN ((size_t)UINT64_C(0x80000000000)) */
#define MAXALIGN ((size_t)0x2000000LU)
#define NITER 4

int
main(void)
{
	int r;
	void *p;
	size_t nsz, rsz, sz, alignment, total;
	unsigned i;
	void *ps[NITER];

	malloc_printf("Test begin\n");

	sz = 42;
	nsz = 0;
	r = nallocm(&nsz, sz, 0);
	if (r != ALLOCM_SUCCESS) {
		malloc_printf("Unexpected nallocm() error\n");
		abort();
	}
	rsz = 0;
	r = allocm(&p, &rsz, sz, 0);
	if (r != ALLOCM_SUCCESS) {
		malloc_printf("Unexpected allocm() error\n");
		abort();
	}
	if (rsz < sz)
		malloc_printf("Real size smaller than expected\n");
	if (nsz != rsz)
		malloc_printf("nallocm()/allocm() rsize mismatch\n");
	if (dallocm(p, 0) != ALLOCM_SUCCESS)
		malloc_printf("Unexpected dallocm() error\n");

	r = allocm(&p, NULL, sz, 0);
	if (r != ALLOCM_SUCCESS) {
		malloc_printf("Unexpected allocm() error\n");
		abort();
	}
	if (dallocm(p, 0) != ALLOCM_SUCCESS)
		malloc_printf("Unexpected dallocm() error\n");

	nsz = 0;
	r = nallocm(&nsz, sz, ALLOCM_ZERO);
	if (r != ALLOCM_SUCCESS) {
		malloc_printf("Unexpected nallocm() error\n");
		abort();
	}
	rsz = 0;
	r = allocm(&p, &rsz, sz, ALLOCM_ZERO);
	if (r != ALLOCM_SUCCESS) {
		malloc_printf("Unexpected allocm() error\n");
		abort();
	}
	if (nsz != rsz)
		malloc_printf("nallocm()/allocm() rsize mismatch\n");
	if (dallocm(p, 0) != ALLOCM_SUCCESS)
		malloc_printf("Unexpected dallocm() error\n");

#if LG_SIZEOF_PTR == 3
	alignment = UINT64_C(0x8000000000000000);
	sz        = UINT64_C(0x8000000000000000);
#else
	alignment = 0x80000000LU;
	sz        = 0x80000000LU;
#endif
	nsz = 0;
	r = nallocm(&nsz, sz, ALLOCM_ALIGN(alignment));
	if (r == ALLOCM_SUCCESS) {
		malloc_printf(
		    "Expected error for nallocm(&nsz, %zu, %#x)\n",
		    sz, ALLOCM_ALIGN(alignment));
	}
	rsz = 0;
	r = allocm(&p, &rsz, sz, ALLOCM_ALIGN(alignment));
	if (r == ALLOCM_SUCCESS) {
		malloc_printf(
		    "Expected error for allocm(&p, %zu, %#x)\n",
		    sz, ALLOCM_ALIGN(alignment));
	}
	if (nsz != rsz)
		malloc_printf("nallocm()/allocm() rsize mismatch\n");

#if LG_SIZEOF_PTR == 3
	alignment = UINT64_C(0x4000000000000000);
	sz        = UINT64_C(0x8400000000000001);
#else
	alignment = 0x40000000LU;
	sz        = 0x84000001LU;
#endif
	nsz = 0;
	r = nallocm(&nsz, sz, ALLOCM_ALIGN(alignment));
	if (r != ALLOCM_SUCCESS)
		malloc_printf("Unexpected nallocm() error\n");
	rsz = 0;
	r = allocm(&p, &rsz, sz, ALLOCM_ALIGN(alignment));
	if (r == ALLOCM_SUCCESS) {
		malloc_printf(
		    "Expected error for allocm(&p, %zu, %#x)\n",
		    sz, ALLOCM_ALIGN(alignment));
	}

	alignment = 0x10LU;
#if LG_SIZEOF_PTR == 3
	sz = UINT64_C(0xfffffffffffffff0);
#else
	sz = 0xfffffff0LU;
#endif
	nsz = 0;
	r = nallocm(&nsz, sz, ALLOCM_ALIGN(alignment));
	if (r == ALLOCM_SUCCESS) {
		malloc_printf(
		    "Expected error for nallocm(&nsz, %zu, %#x)\n",
		    sz, ALLOCM_ALIGN(alignment));
	}
	rsz = 0;
	r = allocm(&p, &rsz, sz, ALLOCM_ALIGN(alignment));
	if (r == ALLOCM_SUCCESS) {
		malloc_printf(
		    "Expected error for allocm(&p, %zu, %#x)\n",
		    sz, ALLOCM_ALIGN(alignment));
	}
	if (nsz != rsz)
		malloc_printf("nallocm()/allocm() rsize mismatch\n");

	for (i = 0; i < NITER; i++)
		ps[i] = NULL;

	for (alignment = 8;
	    alignment <= MAXALIGN;
	    alignment <<= 1) {
		total = 0;
		malloc_printf("Alignment: %zu\n", alignment);
		for (sz = 1;
		    sz < 3 * alignment && sz < (1U << 31);
		    sz += (alignment >> (LG_SIZEOF_PTR-1)) - 1) {
			for (i = 0; i < NITER; i++) {
				nsz = 0;
				r = nallocm(&nsz, sz,
				    ALLOCM_ALIGN(alignment) | ALLOCM_ZERO);
				if (r != ALLOCM_SUCCESS) {
					malloc_printf(
					    "nallocm() error for size %zu"
					    " (%#zx): %d\n",
					    sz, sz, r);
					exit(1);
				}
				rsz = 0;
				r = allocm(&ps[i], &rsz, sz,
				    ALLOCM_ALIGN(alignment) | ALLOCM_ZERO);
				if (r != ALLOCM_SUCCESS) {
					malloc_printf(
					    "allocm() error for size %zu"
					    " (%#zx): %d\n",
					    sz, sz, r);
					exit(1);
				}
				if (rsz < sz) {
					malloc_printf(
					    "Real size smaller than"
					    " expected\n");
				}
				if (nsz != rsz) {
					malloc_printf(
					    "nallocm()/allocm() rsize"
					    " mismatch\n");
				}
				if ((uintptr_t)p & (alignment-1)) {
					malloc_printf(
					    "%p inadequately aligned for"
					    " alignment: %zu\n", p, alignment);
				}
				sallocm(ps[i], &rsz, 0);
				total += rsz;
				if (total >= (MAXALIGN << 1))
					break;
			}
			for (i = 0; i < NITER; i++) {
				if (ps[i] != NULL) {
					dallocm(ps[i], 0);
					ps[i] = NULL;
				}
			}
		}
	}

	malloc_printf("Test end\n");
	return (0);
}
