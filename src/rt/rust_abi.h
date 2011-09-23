// ABI-specific routines.

#ifndef RUST_ABI_H
#define RUST_ABI_H

#include <cstdlib>
#include <string>
#include <vector>
#include <stdint.h>

#ifdef __WIN32__
#include <windows.h>
#else
#include <dlfcn.h>
#endif

template<typename T>
class weak_symbol {
private:
    bool init;
    T *data;
    const char *name;

    void fill() {
        if (init)
            return;

#ifdef __WIN32__
        data = (T *)GetProcAddress(GetModuleHandle(NULL), name);
#else
        data = (T *)dlsym(RTLD_DEFAULT, name);
#endif

        init = true;
    }

public:
    weak_symbol(const char *in_name)
    : init(false), data(NULL), name(in_name) {}

    T *&operator*() { fill(); return data; }
};

namespace stack_walk {

struct frame {
    uint8_t *bp;    // The frame pointer.
    void (*ra)();   // The return address.

    frame(void *in_bp, void (*in_ra)()) : bp((uint8_t *)in_bp), ra(in_ra) {}

    inline void next() {
        ra = *(void (**)())(bp + sizeof(void *));
        bp = *(uint8_t **)bp;
    }

    std::string symbol() const;
};

std::vector<frame> backtrace();
std::string symbolicate(const std::vector<frame> &frames);

}   // end namespace stack_walk


uint32_t get_abi_version();

#endif

