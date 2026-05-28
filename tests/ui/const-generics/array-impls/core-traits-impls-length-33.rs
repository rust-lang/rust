//@ check-pass

pub fn yes_as_ref() -> impl AsRef<[u8]> {
    [0; 33]
}

pub fn yes_as_mut() -> impl AsMut<[u8]> {
    [0; 33]
}

pub fn yes_borrow() -> impl std::borrow::Borrow<[u8]> {
    [0; 33]
}

pub fn yes_borrow_mut() -> impl std::borrow::BorrowMut<[u8]> {
    [0; 33]
}

pub fn yes_try_from_slice() -> impl std::convert::TryFrom<&'static [u8]> {
    [0; 33]
}

pub fn yes_ref_try_from_slice() -> impl std::convert::TryFrom<&'static [u8]> {
    let a: &'static _ = &[0; 33];
    a
}

pub fn yes_hash() -> impl std::hash::Hash {
    [0; 33]
}

pub fn yes_debug() -> impl std::fmt::Debug {
    [0; 33]
}

pub fn yes_ref_into_iterator() -> impl IntoIterator<Item=&'static u8> {
    let a: &'static _ = &[0; 33];
    a
}

pub fn yes_partial_eq() -> impl PartialEq<[u8; 33]> {
    [0; 33]
}

pub fn yes_partial_eq_slice() -> impl PartialEq<[u8]> {
    [0; 33]
}

pub fn yes_slice_partial_eq() -> impl PartialEq<[u8; 33]> {
    let a: &'static _ = &[0; 33];
    &a[..]
}

pub fn yes_eq() -> impl Eq {
    [0; 33]
}

pub fn yes_partial_ord() -> impl PartialOrd<[u8; 33]> {
    [0; 33]
}

pub fn yes_ord() -> impl Ord {
    [0; 33]
}

fn main() {}
