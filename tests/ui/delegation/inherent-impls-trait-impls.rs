#![feature(fn_delegation)]

trait Trait {
    fn foo(&self) {}
}

struct X;
impl Trait for X {
    fn foo(&self) {}
}

reuse X::foo;
//~^ ERROR: no associated function or constant named `foo` found for struct `X` in the current scope

fn main() {
    foo();
}
