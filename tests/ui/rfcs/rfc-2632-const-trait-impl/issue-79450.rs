//@ compile-flags: -Znext-solver
#![allow(incomplete_features)]
#![feature(const_trait_impl, effects)]

#[const_trait]
trait Tr {
    fn req(&self);

    fn prov(&self) {
        println!("lul"); //~ ERROR: cannot call non-const fn `_print` in constant functions
        self.req();
    }
}

struct S;

impl const Tr for S {
    fn req(&self) {}
}

fn main() {}
