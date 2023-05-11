#![crate_type = "lib"]

fn foo<T, U>(_: U) {}

fn bar() {
    foo(|| {}); //~ ERROR type annotations needed
}
