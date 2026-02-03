//@ check-pass

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

#[type_const]
const N: usize = 2;

trait CollectArray<A> {
    fn inner_array(&mut self) -> [A; N];
}

fn main() {}
