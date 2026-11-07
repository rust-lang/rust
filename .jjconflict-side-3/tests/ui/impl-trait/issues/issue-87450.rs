fn bar() -> impl Fn() {
    wrap(wrap(wrap(wrap(foo()))))
}

fn foo() -> impl Fn() {
    //~^ WARN function cannot return without recursing
    //~| ERROR cannot resolve opaque type
    wrap(wrap(wrap(wrap(wrap(wrap(wrap(foo())))))))
}

fn wrap(f: impl Fn()) -> impl Fn() {
    move || f()
}

fn main() {
}
