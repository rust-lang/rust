#![allow(unused)]
#![warn(clippy::cfg_not_test)]

fn important_check() {}

fn main() {
    // Statement
    #[cfg(not(test))]
    //~^ cfg_not_test
    let answer = 42;

    // Expression
    #[cfg(not(test))]
    //~^ cfg_not_test
    important_check();

    // Make sure only not(test) are checked, not other attributes
    #[cfg(not(foo))]
    important_check();
}

#[cfg(not(not(test)))]
struct CfgNotTest;

// Deeply nested `not(test)`
#[cfg(not(test))]
//~^ cfg_not_test
fn foo() {}
#[cfg(all(debug_assertions, not(test)))]
//~^ cfg_not_test
fn bar() {}
#[cfg(not(any(not(debug_assertions), test)))]
//~^ cfg_not_test
fn baz() {}

#[cfg(test)]
mod tests {}
