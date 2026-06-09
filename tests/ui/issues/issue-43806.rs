//@ check-pass

#![deny(unused_results)]

enum Void {}

fn foo() {}

fn bar() -> ! {
    loop {}
}

fn baz() -> Void {
    loop {}
}

fn qux() {
    foo();
    bar();
    baz();
}

fn main() {}
