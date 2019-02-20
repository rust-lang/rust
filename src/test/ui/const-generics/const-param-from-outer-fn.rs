#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

fn foo<const X: u32>() {
    fn bar() -> u32 {
        X //~ ERROR can't use generic parameters from outer function
    }
}

fn main() {}
