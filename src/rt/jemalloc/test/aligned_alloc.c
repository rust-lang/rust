#define	JEMALLOC_MANGLE
#include "jemalloc_test.h"

#define CHUNK 0x400000
/* #define MAXALIGN ((size_t)UINT64_C(0x80000000000)) */
#define MAXALIGN ((size_t)0x2000000LU)
#define NITER 4

int
main(void)
{
	size_t alignment, size, total;
	unsigned i;
	void *p, *ps[NITER];

	malloc_printf("Test begin\n");

	/* Test error conditions. */
	alignment = 0;
	set_errno(0);
	p = aligned_alloc(alignment, 1);
	if (p != NULL || get_errno() != EINVAL) {
		malloc_printf(
		    "Expected error for invalid alignment %zu\n", alignment);
	}

	for (alignment = sizeof(size_t); alignment < MAXALIGN;
	    alignment <<= 1) {
		set_errno(0);
		p = aligned_alloc(alignment + 1, 1);
		if (p != NULL || get_errno() != EINVAL) {
			malloc_printf(
			    "Expected error for invalid alignment %zu\n",
			    alignment + 1);
		}
	}

#if LG_SIZEOF_PTR == 3
	alignment = UINT64_C(0x8000000000000000);
	size      = UINT64_C(0x8000000000000000);
#else
	alignment = 0x80000000LU;
	size      = 0x80000000LU;
#endif
	set_errno(0);
	p = aligned_alloc(alignment, size);
	if (p != NULL || get_errno() != ENOMEM) {
		malloc_printf(
		    "Expected error for aligned_alloc(%zu, %zu)\n",
		    alignment, size);
	}

#if LG_SIZEOF_PTR == 3
	alignment = UINT64_C(0x4000000000000000);
	size      = UINT64_C(0x8400000000000001);
#else
	alignment = 0x40000000LU;
	size      = 0x84000001LU;
#endif
	set_errno(0);
	p = aligned_alloc(alignment, size);
	if (p != NULL || get_errno() != ENOMEM) {
		malloc_printf(
		    "Expected error for aligned_alloc(%zu, %zu)\n",
		    alignment, size);
	}

	alignment = 0x10LU;
#if LG_SIZEOF_PTR == 3
	size = UINT64_C(0xfffffffffffffff0);
#else
	size = 0xfffffff0LU;
#endif
	set_errno(0);
	p = aligned_alloc(alignment, size);
	if (p != NULL || get_errno() != ENOMEM) {
		malloc_printf(
		    "Expected error for aligned_alloc(&p, %zu, %zu)\n",
		    alignment, size);
	}

	for (i = 0; i < NITER; i++)
		ps[i] = NULL;

	for (alignment = 8;
	    alignment <= MAXALIGN;
	    alignment <<= 1) {
		total = 0;
		malloc_printf("Alignment: %zu\n", alignment);
		for (size = 1;
		    size < 3 * alignment && size < (1U << 31);
		    size += (alignment >> (LG_SIZEOF_PTR-1)) - 1) {
			for (i = 0; i < NITER; i++) {
				ps[i] = aligned_alloc(alignment, size);
				if (ps[i] == NULL) {
					char buf[BUFERROR_BUF];

					buferror(buf, sizeof(buf));
					malloc_printf(
					    "Error for size %zu (%#zx): %s\n",
					    size, size, buf);
					exit(1);
				}
				total += malloc_usable_size(ps[i]);
				if (total >= (MAXALIGN << 1))
					break;
			}
			for (i = 0; i < NITER; i++) {
				if (ps[i] != NULL) {
					free(ps[i]);
					ps[i] = NULL;
				}
			}
		}
	}

	malloc_printf("Test end\n");
	return (0);
}
