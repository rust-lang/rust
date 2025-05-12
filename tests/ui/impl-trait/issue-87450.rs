fn bar() -> impl Fn() {
    wrap(wrap(wrap(wrap(foo()))))
}

fn foo() -> impl Fn() {
    //~^ WARNING 5:1: 5:22: function cannot return without recursing [unconditional_recursion]
    //~| ERROR 5:13: 5:22: cannot resolve opaque type [E0720]
    wrap(wrap(wrap(wrap(wrap(wrap(wrap(foo())))))))
}

fn wrap(f: impl Fn()) -> impl Fn() {
    move || f()
}

fn main() {
}
