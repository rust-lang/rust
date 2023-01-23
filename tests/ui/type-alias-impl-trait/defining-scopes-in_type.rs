#![feature(type_alias_impl_trait)]

// check that we don't allow defining hidden types of TAITs in
// const generic defaults.
type Foo = impl Send;

#[rustfmt::skip]
struct Struct<
    const C: usize = {
        let _: Foo = ();
        //~^ ERROR: mismatched types
        0
    },
>;

fn main() {}
