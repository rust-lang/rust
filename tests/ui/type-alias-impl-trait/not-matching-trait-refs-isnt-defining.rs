#![feature(impl_trait_in_assoc_type)]

trait Foo<T> {
    type Assoc;

    fn test() -> u32;
}

struct DefinesOpaque;
impl Foo<DefinesOpaque> for () {
    type Assoc = impl Sized;

    // This test's return type is `u32`, *not* the opaque that is defined above.
    // Previously we were only checking that the self type of the assoc matched,
    // but this doesn't account for other impls with different trait substs.
    fn test() -> <() as Foo<NoOpaques>>::Assoc {
        let _: <Self as Foo<DefinesOpaque>>::Assoc = "";
        //~^ ERROR mismatched types

        1
    }
}

struct NoOpaques;
impl Foo<NoOpaques> for () {
    type Assoc = u32;

    fn test() -> u32 {
        1
    }
}

fn main() {}
