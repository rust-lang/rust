// regression test for rust-lang/rust#67883

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

struct Caca<const A: &'static str> {
    s: String
}

impl<const A: &str> Default for Caca<A> {
    //~^ ERROR: lifetime of reference outlives
    fn default() -> Self {
        let a = Self::A; //~ ERROR: no associated item named `A` found
        Self {
            s: A.to_string()
        }
    }
}

fn main() {
    println!("Hello, world!");
}
