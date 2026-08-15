#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "bar.h"

extern void foo(char*);

void bar(char *ptr) {
    if (((uintptr_t)ptr >> 56) != 0x2f) {
        fprintf(stderr, "Top byte corrupted on Rust -> C FFI boundary!\n");
        exit(1);
    }

    if (strcmp(ptr, "cd")) {
        fprintf(stderr, "invalid data in memory; expected 'cd', got '%s'\n", ptr);
        exit(1);
    }
}

int main(void)
{
    // Construct a pointer with an arbitrary tag in bits 56-59, simulating an MTE tag.
    // It's only necessary that the tag is preserved across FFI bounds for this test.
    char *ptr;

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

    ptr[0] = 'a';
    ptr[1] = 'b';
    ptr[2] = '\0';

    foo(ptr);

    return 0;
}
