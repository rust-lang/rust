#ifndef __BAR_H
#define __BAR_H

#include <sys/mman.h>
#include <sys/auxv.h>
#include <sys/prctl.h>
#include <unistd.h>
#include <stdio.h>

// Set the allocation tag on the destination address using the STG instruction.
#define set_tag(tagged_addr) do {                                      \
    asm volatile("stg %0, [%0]" : : "r" (tagged_addr) : "memory"); \
} while (0)

int mte_enabled() {
    return (getauxval(AT_HWCAP2)) & HWCAP2_MTE;
}

void *alloc_page() {
    // Enable MTE with synchronous checking
    if (prctl(PR_SET_TAGGED_ADDR_CTRL,
              PR_TAGGED_ADDR_ENABLE | PR_MTE_TCF_SYNC | (0xfffe << PR_MTE_TAG_SHIFT),
              0, 0, 0))
    {
        perror("prctl() failed");
    }

    // Using `mmap` allows us to ensure that, on systems which support MTE, the allocated
    // memory is 16-byte aligned for MTE.
    // This also allows us to explicitly specify whether the region should be protected by
    // MTE or not.
    if (mte_enabled()) {
        void *ptr = mmap(NULL, sysconf(_SC_PAGESIZE),
                         PROT_READ | PROT_WRITE | PROT_MTE, MAP_PRIVATE | MAP_ANONYMOUS,
                         -1, 0);
    } else {
        void *ptr = mmap(NULL, sysconf(_SC_PAGESIZE),
                         PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                         -1, 0);
    }
}

#endif // __BAR_H
