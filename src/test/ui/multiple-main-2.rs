#![feature(main)]

#[main]
fn bar() {
}

#[main]
fn foo() { //~ ERROR multiple functions with a #[main] attribute
}
