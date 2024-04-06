fn foo() {
    builtin#asm(0);
    builtin#format_args("", 0, 1, a = 2 + 3, a + b);
    builtin#offset_of(Foo, bar.baz.0);
}
