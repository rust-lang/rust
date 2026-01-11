#![feature(generic_const_items, min_generic_const_args)]
#![expect(incomplete_features)]

trait Foo {
    #[type_const]
    const ASSOC<const N: u32>: u32;
}

impl Foo for () {
    #[type_const]
    const ASSOC<const N: u32>: u32 = N;
}

fn bar<const N: u64, T: Foo<ASSOC<N> = { N }>>() {}
//~^ ERROR: the constant `N` is not of type `u32`

fn main() {
    bar::<10_u64, ()>();
    //~^ ERROR: the constant `10` is not of type `u32`
}
