// check-pass

#![feature(c_unwind, const_extern_fn)]

// We don't unwind in const-eval anyways.
const extern "C" fn foo() {
    panic!()
}

const fn bar() {
    foo();
}

fn main() {
    bar();
}
