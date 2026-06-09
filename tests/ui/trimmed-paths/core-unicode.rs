//@ edition: 2024

// Test that the `#[doc(hidden)]` module `core::unicode` module does not
// disqualify another item named `unicode` from path trimming in diagnostics.

use core::marker::PhantomData;

mod inner {
    #[expect(non_camel_case_types)]
    pub(crate) enum unicode {}
}

fn main() {
    let PhantomData::<(inner::unicode, u32)> = PhantomData::<(u32, inner::unicode)>;
    //~^ ERROR mismatched types [E0308]
    //~| NOTE expected `PhantomData<(u32, unicode)>`, found `PhantomData<(unicode, u32)>`
    //~| NOTE this expression has type `PhantomData<(u32, unicode)>`
    //~| NOTE expected struct `PhantomData<(u32, unicode)>`
}
