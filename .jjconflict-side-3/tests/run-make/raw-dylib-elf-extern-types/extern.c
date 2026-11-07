typedef struct FOO {
    char val;
} FOO;

FOO* create_foo() {
    static FOO foo;
    return &foo;
}

void set_foo(FOO* foo, char val) {
    foo->val = val;
}

char get_foo(FOO* foo) {
    return foo->val;
}
