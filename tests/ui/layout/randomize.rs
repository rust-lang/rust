//@ run-pass
//@ revisions: normal randomize-layout
//@ [randomize-layout]compile-flags: -Zrandomize-layout -Zlayout-seed=2

#![feature(offset_of_enum)]

use std::ptr;


// these types only have their field offsets taken, they're never constructed
#[allow(dead_code)]
pub struct Foo<T>(u32, T, u8);
#[allow(dead_code)]
pub struct Wrapper<T>(T);
#[repr(transparent)]
#[allow(dead_code)]
pub struct TransparentWrapper(u16);

const _: () = {
    // Behavior of the current non-randomized implementation, not guaranteed
    #[cfg(not(randomize_layout))]
    assert!(std::mem::offset_of!(Foo::<u16>, 1) == std::mem::offset_of!(Foo::<Wrapper<u16>>, 1));

    // under randomization Foo<T> != Foo<U>
    #[cfg(randomize_layout)]
    assert!(std::mem::offset_of!(Foo::<u16>, 1) != std::mem::offset_of!(Foo::<Wrapper<u16>>, 1));

    // Even transparent wrapper inner types get a different layout since associated type
    // specialization could result in the outer type behaving differently depending on the exact
    // inner type.
    #[cfg(randomize_layout)]
    assert!(
        std::mem::offset_of!(Foo::<u16>, 1) != std::mem::offset_of!(Foo::<TransparentWrapper>, 1)
    );

    // Currently all fn pointers are treated interchangably even with randomization. Not guaranteed.
    // Associated type specialization could also break this.
    assert!(
        std::mem::offset_of!(Foo::<fn(u32)>, 1) == std::mem::offset_of!(Foo::<fn() -> usize>, 1)
    );

    // But subtype coercions must always result in the same layout.
    assert!(
        std::mem::offset_of!(Foo::<fn(&u32)>, 1) == std::mem::offset_of!(Foo::<fn(&'static u32)>, 1)
    );

    // Randomization must uphold NPO guarantees
    assert!(std::mem::offset_of!(Option::<&usize>, Some.0) == 0);
    assert!(std::mem::offset_of!(Result::<&usize, ()>, Ok.0) == 0);
};

#[allow(dead_code)]
struct Unsizable<T: ?Sized>(usize, T);

fn main() {
    // offset_of doesn't let us probe the unsized field, check at runtime.
    let x = &Unsizable::<[u32; 4]>(0, [0; 4]);
    let y: &Unsizable::<[u32]> = x;

    // type coercion must not change the layout.
    assert_eq!(ptr::from_ref(&x.1).addr(), ptr::from_ref(&y.1).addr());
}
