//@ run-pass
// Issue #14660


mod bleh {
    macro_rules! foo {
        () => {
            pub fn bar() { }
        }
    }

    foo!();
}

fn main() {
    bleh::bar();
}
