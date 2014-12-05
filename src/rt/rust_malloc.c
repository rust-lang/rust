#include <stddef.h>

void *je_malloc(size_t size);
void *je_calloc(size_t num, size_t size);
int je_posix_memalign(void **memptr, size_t alignment, size_t size);
void *je_aligned_alloc(size_t alignment, size_t size);
void *je_realloc(void *ptr, size_t size);
void je_free(void *ptr);

void *je_mallocx(size_t size, int flags);
void *je_rallocx(void *ptr, size_t size, int flags);
size_t je_xallocx(void *ptr, size_t size, size_t extra, int flags);
size_t je_sallocx(const void *ptr, int flags);
void je_dallocx(void *ptr, int flags);
void je_sdallocx(void *ptr, size_t size, int flags);
size_t je_nallocx(size_t size, int flags);

int je_mallctl(const char *name, void *oldp, size_t *oldlenp, void *newp, size_t newlen);
int je_mallctlnametomib(const char *name, size_t *mibp, size_t *miblenp);
int je_mallctlbymib(const size_t *mib, size_t miblen, void *oldp, size_t *oldlenp, void *newp,
                    size_t newlen);
void je_malloc_stats_print(void (*write_cb)(void *, const char *), void *je_cbopaque,
                           const char *opts);
size_t je_malloc_usable_size(const void *ptr);

void *je_memalign(size_t alignment, size_t size);
#if !defined(__ANDROID__)
void *je_valloc(size_t size);
#endif

void *malloc(size_t size) {
    return je_malloc(size);
}

void *calloc(size_t num, size_t size) {
    return je_calloc(num, size);
}

int posix_memalign(void **memptr, size_t alignment, size_t size) {
    return je_posix_memalign(memptr, alignment, size);
}

void *aligned_alloc(size_t alignment, size_t size) {
    return je_aligned_alloc(alignment, size);
}

void *realloc(void *ptr, size_t size) {
    return je_realloc(ptr, size);
}

void free(void *ptr) {
    je_free(ptr);
}

void *mallocx(size_t size, int flags) {
    return je_mallocx(size, flags);
}

void *rallocx(void *ptr, size_t size, int flags) {
    return je_rallocx(ptr, size, flags);
}

size_t xallocx(void *ptr, size_t size, size_t extra, int flags) {
    return je_xallocx(ptr, size, extra, flags);
}

size_t sallocx(const void *ptr, int flags) {
    return je_sallocx(ptr, flags);
}

void dallocx(void *ptr, int flags) {
    je_dallocx(ptr, flags);
}

void sdallocx(void *ptr, size_t size, int flags) {
    je_sdallocx(ptr, size, flags);
}

size_t nallocx(size_t size, int flags) {
    return je_nallocx(size, flags);
}

int mallctl(const char *name, void *oldp, size_t *oldlenp, void *newp, size_t newlen) {
    return je_mallctl(name, oldp, oldlenp, newp, newlen);
}

int mallctlnametomib(const char *name, size_t *mibp, size_t *miblenp) {
    return je_mallctlnametomib(name, mibp, miblenp);
}

int mallctlbymib(const size_t *mib, size_t miblen, void *oldp, size_t *oldlenp, void *newp,
                    size_t newlen) {
    return je_mallctlbymib(mib, miblen, oldp, oldlenp, newp, newlen);
}

void malloc_stats_print(void (*write_cb)(void *, const char *), void *je_cbopaque,
                           const char *opts) {
    return je_malloc_stats_print(write_cb, je_cbopaque, opts);
}

size_t malloc_usable_size(const void *ptr) {
    return je_malloc_usable_size(ptr);
}

void *memalign(size_t alignment, size_t size) {
    return je_memalign(alignment, size);
}

void *valloc(size_t size) {
    return je_valloc(size);
}
