#![feature(const_trait_impl)]

const trait Tr {
    fn req(&self);

    fn default() {}
}

struct S;

const impl Tr for u16 {
    fn default() {}
} //~^^ ERROR not all trait items implemented

fn main() {}
