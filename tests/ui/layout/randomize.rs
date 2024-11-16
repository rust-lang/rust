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

    // but repr(transparent) should make them the same again.
    // maybe not strictly guaranteed? but UCG has been leaning in that direction at least
    #[cfg(randomize_layout)]
    assert!(
        std::mem::offset_of!(Foo::<u16>, 1) == std::mem::offset_of!(Foo::<TransparentWrapper>, 1)
    );
};
