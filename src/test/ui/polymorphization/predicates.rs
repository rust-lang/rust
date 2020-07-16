// build-fail
#![feature(rustc_attrs)]

// This test checks that `T` is considered used in `foo`, because it is used in a predicate for
// `I`, which is used.

#[rustc_polymorphize_error]
fn bar<I>() {
    //~^ ERROR item has unused generic parameters
}

#[rustc_polymorphize_error]
fn foo<I, T>(_: I)
where
    I: Iterator<Item = T>,
{
    bar::<I>()
}

fn main() {
    let x = &[2u32];
    foo(x.iter());
}
