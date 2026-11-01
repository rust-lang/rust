//@no-rustfix
#![warn(clippy::needless_type_cast)]

struct Foo(*mut core::ffi::c_void);

enum Bar {
    Variant(*mut core::ffi::c_void),
}

// Suggestions will not compile directly, as `123` is a literal which
// is not compatible with the suggested `*mut core::ffi::c_void` type
fn issue_16243() {
    let underlying: isize = 123;
    //~^ needless_type_cast
    let handle: Foo = Foo(underlying as _);

    let underlying: isize = 123;
    //~^ needless_type_cast
    let handle: Bar = Bar::Variant(underlying as _);
}
