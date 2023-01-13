#include <cstdio>

// Since this is a global variable, its constructor will be called before
// main() is executed. But only if the object file containing it actually
// gets linked into the executable.
struct Foo {
    Foo() {
        printf("static-initializer.");
        fflush(stdout);
    }
} FOO;
