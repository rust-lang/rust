//@ check-fail
//@ compile-flags: -Z tiny-const-eval-limit

const fn foo() {}

const fn call_foo() -> u32 {
    foo();
    foo();
    foo();
    foo();
    foo();

    foo();
    foo();
    foo();
    foo();
    foo();

    foo();
    foo();
    foo();
    foo();
    foo();

    foo();
    foo();
    foo();
    foo(); //~ ERROR is taking a long time
    0
}

const X: u32 = call_foo();

fn main() {
    println!("{X}");
}
