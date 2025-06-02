#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "bar.h"

extern void foo(unsigned int *);

void bar(unsigned int *ptr) {
    if (((uintptr_t)ptr >> 56) != 0x1f) {
        fprintf(stderr, "Top byte corrupted on Rust -> C FFI boundary!\n");
        exit(1);
    }
}

int main(void)
{
    // Construct a pointer with an arbitrary tag in bits 56-59, simulating an MTE tag.
    // It's only necessary that the tag is preserved across FFI bounds for this test.
    unsigned int *ptr;

    ptr = alloc_page();
    if (ptr == MAP_FAILED)
    {
        perror("mmap() failed");
        return EXIT_FAILURE;
    }

    // Store an arbitrary tag in bits 56-59 of the pointer (where an MTE tag may be),
    // and a different value in the ignored top 4 bits.
    ptr = (unsigned int *)((uintptr_t)ptr | 0x1fl << 56);

    if (mte_enabled()) {
        set_tag(ptr);
    }

    ptr[0] = 61;
    ptr[1] = 62;

    foo(ptr); // should change the contents of the page to start with 0x63 0x64 and call `bar`

    if (ptr[0] != 0x63 || ptr[1] != 0x64) {
        fprintf(stderr, "invalid data in memory; expected '63 64', got '%d %d'\n", ptr[0], ptr[1]);
        return EXIT_FAILURE;
    }

    return 0;
}
