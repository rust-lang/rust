//@ run-pass
#![allow(unused)]
#![feature(f128)]
#![feature(f16)]

// Same as the feature gate tests but ensure we can use the types
mod check_f128 {
    const A: f128 = 10.0;

    pub fn foo() {
        let a: f128 = 100.0;
        let b = 0.0f128;
        bar(1.23);
    }

    fn bar(a: f128) {}

    struct Bar {
        a: f128,
    }
}

mod check_f16 {
    const A: f16 = 10.0;

    pub fn foo() {
        let a: f16 = 100.0;
        let b = 0.0f16;
        bar(1.23);
    }

    fn bar(a: f16) {}

    struct Bar {
        a: f16,
    }
}

fn main() {
    check_f128::foo();
    check_f16::foo();
}
