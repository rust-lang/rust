// ignore-license
// Pragma needed cause of gcc bug on windows: http://gcc.gnu.org/bugzilla/show_bug.cgi?id=52991

#include <assert.h>

#ifdef _MSC_VER
#pragma pack(push,1)
struct Foo {
    char a;
    short b;
    char c;
};
#else
#pragma pack(1)
struct __attribute__((packed)) Foo {
    char a;
    short b;
    char c;
};
#endif

struct Foo foo(struct Foo foo) {
    assert(foo.a == 1);
    assert(foo.b == 2);
    assert(foo.c == 3);
    return foo;
}
