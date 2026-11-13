//@ check-pass

#![feature(type_alias_impl_trait)]

type Closure = impl Fn(u32) -> u32;

#[define_opaque(Closure)]
const ADDER: Closure = |x| x + 1;

fn main() {
    let z = (ADDER)(1);
}
