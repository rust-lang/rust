//@ build-pass
//@ revisions: normal randomize-layout
//@ [randomize-layout]compile-flags: -Zrandomize-layout

#![crate_type = "lib"]

struct Foo<T>(u32, T, u8);

struct Wrapper<T>(T);

#[repr(transparent)]
struct TransparentWrapper(u16);

const _: () = {
    // behavior of the current implementation, not guaranteed
    #[cfg(not(randomize_layout))]
    assert!(std::mem::offset_of!(Foo::<u16>, 1) == std::mem::offset_of!(Foo::<Wrapper<u16>>, 1));

    // under randomization Foo<T> != Foo<U>
    #[cfg(randomize_layout)]
    assert!(std::mem::offset_of!(Foo::<u16>, 1) != std::mem::offset_of!(Foo::<Wrapper<u16>>, 1));

    // Even transparent wrapper inner types get a different layout since associated type
    // pecialization could result in the outer type behaving differently depending on the exact
    // inner type.
    #[cfg(randomize_layout)]
    assert!(
        std::mem::offset_of!(Foo::<u16>, 1) != std::mem::offset_of!(Foo::<TransparentWrapper>, 1)
    );
};
