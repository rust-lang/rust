#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "bar.h"

typedef void (*fp)(int (*)());

extern void foo(fp);

void bar(int (*ptr)()) {
    if (((uintptr_t)ptr >> 56) != 0x2f) {
        fprintf(stderr, "Top byte corrupted on Rust -> C FFI boundary!\n");
        exit(1);
    }

    int r = (*ptr)();
    if (r != 32) {
        fprintf(stderr, "invalid return value; expected 32, got '%d'\n", r);
        exit(1);
    }
}

int main(void)
{
    fp ptr = alloc_page();
    if (ptr == MAP_FAILED)
    {
        perror("mmap() failed");
        return EXIT_FAILURE;
    }

    // Store an arbitrary tag in bits 56-59 of the pointer (where an MTE tag may be),
    // and a different value in the ignored top 4 bits.
    ptr = (fp)((uintptr_t)&bar | 0x1fl << 56);

    foo(ptr);

    return 0;
}
