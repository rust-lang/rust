#![deny(unfulfilled_lint_expectations)]

trait Trait {
    fn method();
}

struct S;

impl Trait for S {
    #[expect(dead_code)] //~ ERROR: this lint expectation is unfulfilled
    fn method() {
        helper();
    }
}

#[expect(dead_code)] //~ ERROR: this lint expectation is unfulfilled
fn helper() {}

fn call<T: Trait>() {
    T::method();
}

fn main() {
    call::<S>();
}
