//@ revisions: normal import_trait_associated_functions
#![cfg_attr(import_trait_associated_functions, feature(import_trait_associated_functions))]

// Makes sure that imported constant can be used in pattern bindings.

use MyDefault::DEFAULT; //[normal]~ ERROR `use` associated items of traits is unstable

trait MyDefault {
    const DEFAULT: Self;
}

impl MyDefault for u32 {
    const DEFAULT: u32 = 0;
}

impl MyDefault for () {
    const DEFAULT: () = ();
}

fn foo(x: u32) -> u32 {
    let DEFAULT: u32 = 0; //~ ERROR refutable pattern in local binding
    const DEFAULT: u32 = 0;
    if let DEFAULT = x { DEFAULT } else { 1 }
}

fn bar() {
    let DEFAULT = ();
}

fn main() {}
