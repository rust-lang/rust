// check-pass

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
    foo();
    foo();
    0
}

const X: u32 = call_foo();

fn main() {
    println!("{X}");
}
