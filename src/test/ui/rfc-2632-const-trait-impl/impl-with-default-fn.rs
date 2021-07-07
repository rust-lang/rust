#![feature(const_trait_impl)]
#![allow(incomplete_features)]

trait Tr {
    fn req(&self);

    fn prov(&self) {
        println!("lul");
        self.req();
    }
}

struct S;

impl const Tr for S {
    fn req(&self) {}
}
//~^^^ ERROR const trait implementations may not use default functions

fn main() {}
