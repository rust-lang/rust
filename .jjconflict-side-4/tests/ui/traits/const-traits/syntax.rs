//@ compile-flags: -Z parse-crate-root-only

#![feature(const_trait_impl)]

// This is going down the slice/array parsing route
impl [const] T {}
//~^ ERROR: expected identifier, found `]`

impl const T {}
