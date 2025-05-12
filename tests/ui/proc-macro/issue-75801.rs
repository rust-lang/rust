//@ proc-macro: issue-75801.rs

// Regression test for #75801.

#[macro_use]
extern crate issue_75801;

macro_rules! foo {
    ($arg:expr) => {
        #[foo]
        fn bar() {
            let _bar: u32 = $arg;
        }
    };
}

foo!("baz"); //~ ERROR: mismatched types [E0308]

fn main() {}
