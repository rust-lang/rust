//! This test checks that ZSTs can be safely initialized from
//! `MaybeUninit::uninit().assume_init()` and `std::mem::uninitialized()`
//! (which is deprecated). This is safe because ZSTs inherently
//! require no actual memory initialization, as they occupy no memory.

//@ build-pass

use std::mem::MaybeUninit;

struct Foo;

#[allow(deprecated)]
pub fn main() {
    unsafe {
        // `Foo` and `[Foo; 2]` are both zero sized and inhabited, so this is safe.
        let _x: Foo = MaybeUninit::uninit().assume_init();
        let _x: [Foo; 2] = MaybeUninit::uninit().assume_init();
        let _x: Foo = std::mem::uninitialized();
        let _x: [Foo; 2] = std::mem::uninitialized();
    }
}
