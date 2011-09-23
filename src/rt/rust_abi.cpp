// ABI-specific routines.

#include <vector>
#include <cstdlib>
#include <stdint.h>
#include "rust_abi.h"

#define END_OF_STACK_RA     (void (*)())0xdeadbeef

weak_symbol<uint32_t> abi_version("rust_abi_version");

uint32_t get_abi_version() {
    return (*abi_version == NULL) ? 0 : **abi_version;
}

namespace stack_walk {

std::vector<frame>
backtrace() {
    std::vector<frame> frames;

    // Ideally we would use the current value of EIP here, but there's no
    // portable way to get that and there are never any GC roots in our C++
    // frames anyhow.
    frame f(__builtin_frame_address(0), (void (*)())NULL);

    while (f.ra != END_OF_STACK_RA) {
        frames.push_back(f);
        f.next();
    }
    return frames;
}

}   // end namespace stack_walk

