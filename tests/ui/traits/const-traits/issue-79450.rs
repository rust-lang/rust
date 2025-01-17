//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

#[const_trait]
trait Tr {
    fn req(&self);

    fn prov(&self) {
        println!("lul"); //~ ERROR: cannot call non-const function `_print` in constant functions
        self.req();
    }
}

struct S;

impl const Tr for S {
    fn req(&self) {}
}

fn main() {}
