// ICE Unexpected unsized type tail: &ReStatic [u8]
// issue: rust-lang/rust#122488
use std::ops::Deref;

struct ArenaSet<U: Deref, V: ?Sized = <U as Deref>::Target>(V, U);
//~^ ERROR the size for values of type `V` cannot be known at compilation time

const DATA: *const ArenaSet<Vec<u8>> = std::ptr::null_mut();
//~^ ERROR the type `ArenaSet<Vec<u8>, [u8]>` has an unknown layout

pub fn main() {}
