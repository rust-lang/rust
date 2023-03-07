// run-pass
// Issue #14660

// pretty-expanded FIXME #23616

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
