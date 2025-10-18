// ICE when applying `#![test]` to the crate root,
// though only when specified with a full path. `#![test]` is not enough.
// Fixes #114920
#![core::prelude::v1::test]
//~^ ERROR inner macro attributes are unstable
//~| ERROR the `#[test]` attribute may only be used on a free function


fn main() {} // not important to reproduce the issue
