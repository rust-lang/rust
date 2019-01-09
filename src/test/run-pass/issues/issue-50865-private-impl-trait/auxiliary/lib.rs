#![crate_type = "lib"]

pub fn bar<P>( // Error won't happen if "bar" is not generic
    _baz: P,
) {
    hide_foo()();
}

fn hide_foo() -> impl Fn() { // Error won't happen if "iterate" hasn't impl Trait or has generics
    foo
}

fn foo() { // Error won't happen if "foo" isn't used in "iterate" or has generics
}
