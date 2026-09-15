//@ check-pass

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

const N: usize = core::direct_const_arg!(2);

trait CollectArray<A> {
    fn inner_array(&mut self) -> [A; N];
}

fn main() {}
