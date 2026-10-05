// This test ensures that we're linking to `alloc` for the `Error::new` method as
// it is defined in `alloc`. The path between `core` (`core::io::error::Error`) and `alloc`
// (`alloc::io::Error`) is different though, so this test ensures that we generate the
// correct one (and also that we link to `std` rather than `alloc`).

#![crate_name = "foo"]

// Needed for the `alloc` imports.
extern crate alloc;

//@ compile-flags: -Zunstable-options --generate-link-to-definition

//@ has 'src/foo/link-to-alloc.rs.html'

use alloc::io;

pub fn foo() -> Option<io::Error> {
    //@ has - '//a[@href="{{channel}}/core/io/error/struct.Error.html"]' 'Error'
    //@ has - '//a[@href="{{channel}}/std/io/struct.Error.html#method.new"]' 'new'
    // This check is just to show that we also link to `core` when relevant.
    //@ has - '//a[@href="{{channel}}/core/io/error/enum.ErrorKind.html#variant.InvalidInput"]' \
    //        'InvalidInput'
    Some(io::Error::new(io::ErrorKind::InvalidInput, "blob"))
}
