//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

#[const_trait]
trait Tr {
    (const) fn req(&self);

    (const) fn prov(&self) {
        println!("lul"); //~ ERROR: cannot call non-const function `_print` in constant functions
        self.req();
    }
}

struct S;

impl const Tr for S {
    (const) fn req(&self) {}
}

fn main() {}
