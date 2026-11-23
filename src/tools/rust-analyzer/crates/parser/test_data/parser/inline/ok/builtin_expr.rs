fn foo() {
    k#asm("");
    k#format_args("", 0, 1, a = 2 + 3, a + b);
    k#offset_of(Foo, bar.baz.0);
    k#cfg(any(true, x = "y"));
}
