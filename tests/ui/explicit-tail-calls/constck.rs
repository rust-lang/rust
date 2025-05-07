#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

const fn f() {
    if false {
        become not_const();
        //~^ error: cannot call non-const function `not_const` in constant functions
    }
}

const fn g((): ()) {
    if false {
        become yes_const(not_const());
        //~^ error: cannot call non-const function `not_const` in constant functions
    }
}

fn not_const() {}

const fn yes_const((): ()) {}

fn main() {}
