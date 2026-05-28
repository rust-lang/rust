//@ run-pass
// Issue #34183

macro_rules! foo {
    () => {
        fn foo() { }
    }
}

macro_rules! bar {
    () => {
        fn bar();
    }
}

trait Bleh {
    foo!();
    bar!();
}

struct Test;

impl Bleh for Test {
    fn bar() {}
}

fn main() {
    Test::bar();
    Test::foo();
}
