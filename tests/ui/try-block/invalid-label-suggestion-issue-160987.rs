//! Regression test for https://github.com/rust-lang/rust/issues/160987.
//! A label suggested for a try block must use valid syntax.

//@ edition: 2024

#![feature(try_blocks)]

fn main() {
    try {
        break;
        //~^ ERROR `break` outside of a loop or labeled block
        None?;
    };
}
