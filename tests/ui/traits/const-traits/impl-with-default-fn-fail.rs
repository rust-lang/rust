#![feature(const_trait_impl)]

#[const_trait]
trait Tr {
    (const) fn req(&self);

    (const) fn default() {}
}

struct S;

impl const Tr for u16 {
    (const) fn default() {}
} //~^^ ERROR not all trait items implemented


fn main() {}
