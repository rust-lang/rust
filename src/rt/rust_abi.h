#ifndef RUST_ABI_H
#define RUST_ABI_H

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

#endif

