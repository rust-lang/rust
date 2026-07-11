//! The aggregate `PartialEq::eq` comparison emitted for constant array/slice
//! patterns must be marked as non-unwinding. If it could unwind, this program
//! would fail borrow-checking in edition 2021: the unwind path from the guard
//! would require the scrutinee temporary, which borrows `referent`, to be
//! dropped in a different order relative to `referent` than on the ordinary
//! path. An explicit `slice == b"ABCD"` guard, which is an unwinding call,
//! still errors here.
//@ check-pass
//@ edition: 2021

struct Referent;
impl Drop for Referent {
    fn drop(&mut self) {}
}

struct DropMeFirst<'a>(&'a Referent);
impl Drop for DropMeFirst<'_> {
    fn drop(&mut self) {}
}

fn foo(slice: &[u8]) -> u32 {
    let referent = Referent;
    match DropMeFirst(&referent) {
        _dropped_first if matches!(slice, b"ABCD") => 0,
        _dropped_first => 1,
    }
}

fn main() {
    assert_eq!(foo(b"ABCD"), 0);
    assert_eq!(foo(b"ZZZZ"), 1);
}
