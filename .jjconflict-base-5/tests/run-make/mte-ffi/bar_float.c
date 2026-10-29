#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "bar.h"

extern void foo(char*);

void bar(char *ptr) {
    if (((uintptr_t)ptr >> 56) != 0x1f) {
        fprintf(stderr, "Top byte corrupted on Rust -> C FFI boundary!\n");
        exit(1);
    }
}

int main(void)
{
    float *ptr = alloc_page();
    if (ptr == MAP_FAILED)
    {
        perror("mmap() failed");
        return EXIT_FAILURE;
    }

    // Store an arbitrary tag in bits 56-59 of the pointer (where an MTE tag may be),
    // and a different value in the ignored top 4 bits.
    ptr = (float *)((uintptr_t)ptr | 0x1fl << 56);

    if (mte_enabled()) {
        set_tag(ptr);
    }

    ptr[0] = 2.0f;
    ptr[1] = 1.5f;

    foo(ptr); // should change the contents of the page and call `bar`

    if (ptr[0] != 0.5f || ptr[1] != 0.2f) {
        fprintf(stderr, "invalid data in memory; expected '0.5 0.2', got '%f %f'\n",
                ptr[0], ptr[1]);
        return EXIT_FAILURE;
    }

    return 0;
}
