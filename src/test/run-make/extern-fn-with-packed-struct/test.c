struct __attribute__((packed)) Foo {
    char a;
    short b;
    char c;
};

struct Foo foo(struct Foo foo) {
    return foo;
}
