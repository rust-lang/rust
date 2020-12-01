// check-pass
// compile-flags: -Z span-debug
// aux-build:test-macros.rs

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;

macro_rules! pass_nonterminal {
    ($line:expr) => {
        #[print_attr_args(a, $line, b)]
        struct S;
    };
}

// `line!()` is not expanded before it's passed to the proc macro.
pass_nonterminal!(line!());

// Test case from #43860.

#[macro_export]
macro_rules! use_contract {
    ($name: ident, $path: expr) => {
        #[derive(Empty)]
        #[empty_helper(path = $path)] // OK
        pub struct $name<T, C> {
            api: T,
            contract: C,
        }
    };
}

use_contract!(ContractName, file!());

fn main() {}
