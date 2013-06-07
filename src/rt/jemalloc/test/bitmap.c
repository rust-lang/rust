#define	JEMALLOC_MANGLE
#include "jemalloc_test.h"

#if (LG_BITMAP_MAXBITS > 12)
#  define MAXBITS	4500
#else
#  define MAXBITS	(1U << LG_BITMAP_MAXBITS)
#endif

static void
test_bitmap_size(void)
{
	size_t i, prev_size;

	prev_size = 0;
	for (i = 1; i <= MAXBITS; i++) {
		size_t size = bitmap_size(i);
		assert(size >= prev_size);
		prev_size = size;
	}
}

static void
test_bitmap_init(void)
{
	size_t i;

	for (i = 1; i <= MAXBITS; i++) {
		bitmap_info_t binfo;
		bitmap_info_init(&binfo, i);
		{
			size_t j;
			bitmap_t *bitmap = malloc(sizeof(bitmap_t) *
				bitmap_info_ngroups(&binfo));
			bitmap_init(bitmap, &binfo);

			for (j = 0; j < i; j++)
				assert(bitmap_get(bitmap, &binfo, j) == false);
			free(bitmap);

		}
	}
}

static void
test_bitmap_set(void)
{
	size_t i;

	for (i = 1; i <= MAXBITS; i++) {
		bitmap_info_t binfo;
		bitmap_info_init(&binfo, i);
		{
			size_t j;
			bitmap_t *bitmap = malloc(sizeof(bitmap_t) *
				bitmap_info_ngroups(&binfo));
			bitmap_init(bitmap, &binfo);

			for (j = 0; j < i; j++)
				bitmap_set(bitmap, &binfo, j);
			assert(bitmap_full(bitmap, &binfo));
			free(bitmap);
		}
	}
}

static void
test_bitmap_unset(void)
{
	size_t i;

	for (i = 1; i <= MAXBITS; i++) {
		bitmap_info_t binfo;
		bitmap_info_init(&binfo, i);
		{
			size_t j;
			bitmap_t *bitmap = malloc(sizeof(bitmap_t) *
				bitmap_info_ngroups(&binfo));
			bitmap_init(bitmap, &binfo);

			for (j = 0; j < i; j++)
				bitmap_set(bitmap, &binfo, j);
			assert(bitmap_full(bitmap, &binfo));
			for (j = 0; j < i; j++)
				bitmap_unset(bitmap, &binfo, j);
			for (j = 0; j < i; j++)
				bitmap_set(bitmap, &binfo, j);
			assert(bitmap_full(bitmap, &binfo));
			free(bitmap);
		}
	}
}

static void
test_bitmap_sfu(void)
{
	size_t i;

	for (i = 1; i <= MAXBITS; i++) {
		bitmap_info_t binfo;
		bitmap_info_init(&binfo, i);
		{
			ssize_t j;
			bitmap_t *bitmap = malloc(sizeof(bitmap_t) *
				bitmap_info_ngroups(&binfo));
			bitmap_init(bitmap, &binfo);

			/* Iteratively set bits starting at the beginning. */
			for (j = 0; j < i; j++)
				assert(bitmap_sfu(bitmap, &binfo) == j);
			assert(bitmap_full(bitmap, &binfo));

			/*
			 * Iteratively unset bits starting at the end, and
			 * verify that bitmap_sfu() reaches the unset bits.
			 */
			for (j = i - 1; j >= 0; j--) {
				bitmap_unset(bitmap, &binfo, j);
				assert(bitmap_sfu(bitmap, &binfo) == j);
				bitmap_unset(bitmap, &binfo, j);
			}
			assert(bitmap_get(bitmap, &binfo, 0) == false);

			/*
			 * Iteratively set bits starting at the beginning, and
			 * verify that bitmap_sfu() looks past them.
			 */
			for (j = 1; j < i; j++) {
				bitmap_set(bitmap, &binfo, j - 1);
				assert(bitmap_sfu(bitmap, &binfo) == j);
				bitmap_unset(bitmap, &binfo, j);
			}
			assert(bitmap_sfu(bitmap, &binfo) == i - 1);
			assert(bitmap_full(bitmap, &binfo));
			free(bitmap);
		}
	}
}

int
main(void)
{
	malloc_printf("Test begin\n");

	test_bitmap_size();
	test_bitmap_init();
	test_bitmap_set();
	test_bitmap_unset();
	test_bitmap_sfu();

	malloc_printf("Test end\n");
	return (0);
}
