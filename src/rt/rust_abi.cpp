// ABI-specific routines.

#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <stdint.h>
#include "rust_abi.h"

#if defined(__APPLE__) || defined(__linux__)
#define HAVE_DLFCN_H
#include <dlfcn.h>
#elif defined(_WIN32)
// TODO
#endif

#define END_OF_STACK_RA     (void (*)())0xdeadbeef

weak_symbol<uint32_t> abi_version("rust_abi_version");

uint32_t get_abi_version() {
    return (*abi_version == NULL) ? 0 : **abi_version;
}

namespace stack_walk {

#ifdef HAVE_DLFCN_H
std::string
frame::symbol() const {
    std::stringstream ss;

    Dl_info info;
    if (!dladdr((void *)ra, &info))
        ss << "??";
    else
        ss << info.dli_sname;

    ss << " @ " << std::hex << (uintptr_t)ra;
    return ss.str();
}
#else
std::string
frame::symbol() {
    std::stringstream ss;
    ss << std::hex << (uintptr_t)ra;
    return ss.str();
}
#endif

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

std::string
symbolicate(const std::vector<frame> &frames) {
    std::stringstream ss;
    std::vector<frame>::const_iterator begin(frames.begin()),
                                       end(frames.end());
    while (begin != end) {
        ss << begin->symbol() << std::endl;
        ++begin;
    }
    return ss.str();
}

}   // end namespace stack_walk

