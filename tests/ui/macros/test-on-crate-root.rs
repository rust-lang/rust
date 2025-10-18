// ICE when applying `#![test]` to the crate root,
// though only when specified with a full path. `#![test]` is not enough.
// Fixes #114920
#![core::prelude::v1::test]



fn main() {} // not important to reproduce the issue
