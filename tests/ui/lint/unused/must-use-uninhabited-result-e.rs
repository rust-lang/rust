// This test checks for uninhabited error type in a result interacts with must
// use lint. Ideally we don't want to lint on `Result<T, !>` as something that
// must be used, because it's equivalent to `T`. However, we need to make sure
// that if there are other reasons for the lint, we still emit it.
//
// See also: <https://github.com/rust-lang/rust/issues/65861>.
//
//@ aux-build:../../../pattern/usefulness/auxiliary/empty.rs
//@ check-pass
#![warn(unused)]
#![feature(never_type)]

extern crate empty;

fn main() {
    Ok::<_, std::convert::Infallible>(());
    Ok::<_, empty::EmptyForeignEnum>(());
    Ok::<_, empty::VisiblyUninhabitedForeignStruct>(());
    Ok::<_, empty::SecretlyUninhabitedForeignStruct>(()); //~ warn: unused `Result` that must be used
    Ok::<_, !>(());

    Ok::<_, !>(Important); //~ warn: unused `Important` that must be used

    very_important(); //~ warn: unused return value of `very_important` that must be used
}

#[must_use]
struct Important;

#[must_use]
fn very_important() -> Result<(), !> { Ok(()) }
