// check-pass

#![feature(unwind_attributes, const_panic)]

// `#[unwind(aborts)]` is okay for a `const fn`. We don't unwind in const-eval anyways.
#[unwind(aborts)]
const fn foo() {
    panic!()
}

const fn bar() {
    foo();
}

fn main() {
    bar();
}
