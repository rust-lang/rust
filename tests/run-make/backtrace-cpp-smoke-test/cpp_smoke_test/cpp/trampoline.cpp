#include <stdio.h>

namespace space {
    template <typename FuncT>
    void templated_trampoline(FuncT func) {
        func();
    }
}

typedef void (*FuncPtr)();

extern "C" void cpp_trampoline(FuncPtr func) {
    space::templated_trampoline(func);
}
