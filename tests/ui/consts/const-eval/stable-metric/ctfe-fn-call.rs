// check-fail
// compile-flags: -Z tiny-const-eval-limit

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
    foo(); //~ ERROR evaluation of constant value failed [E0080]
    0
}

const X: u32 = call_foo();

fn main() {
    println!("{X}");
}
