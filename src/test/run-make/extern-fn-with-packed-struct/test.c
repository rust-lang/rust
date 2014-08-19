// Pragma needed cause of gcc bug on windows: http://gcc.gnu.org/bugzilla/show_bug.cgi?id=52991
#pragma pack(1)
struct __attribute__((packed)) Foo {
    char a;
    short b;
    char c;
};

struct Foo foo(struct Foo foo) {
    return foo;
}
