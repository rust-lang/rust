#![allow(dyn_drop)]

// Check that traitobject 0.1.0 compiles

//! # traitobject
//!
//! Unsafe helpers for working with raw TraitObjects.

/// A trait implemented for all trait objects.
///
/// Implementations for all traits in std are provided.
pub unsafe trait Trait {}

unsafe impl Trait for dyn (::std::any::Any) + Send { }
unsafe impl Trait for dyn (::std::any::Any) + Sync { }
unsafe impl Trait for dyn (::std::any::Any) + Send + Sync { }
unsafe impl<T: ?Sized> Trait for dyn (::std::borrow::Borrow<T>) + Send { }
unsafe impl<T: ?Sized> Trait for dyn (::std::borrow::Borrow<T>) + Sync { }
unsafe impl<T: ?Sized> Trait for dyn (::std::borrow::Borrow<T>) + Send + Sync { }
unsafe impl<T: ?Sized> Trait for dyn (::std::borrow::BorrowMut<T>) + Send { }
unsafe impl<T: ?Sized> Trait for dyn (::std::borrow::BorrowMut<T>) + Sync { }
unsafe impl<T: ?Sized> Trait for dyn (::std::borrow::BorrowMut<T>) + Send + Sync { }
unsafe impl<T: ?Sized> Trait for dyn (::std::convert::AsMut<T>) + Send { }
unsafe impl<T: ?Sized> Trait for dyn (::std::convert::AsMut<T>) + Sync { }
unsafe impl<T: ?Sized> Trait for dyn (::std::convert::AsMut<T>) + Send + Sync { }
unsafe impl<T: ?Sized> Trait for dyn (::std::convert::AsRef<T>) + Send { }
unsafe impl<T: ?Sized> Trait for dyn (::std::convert::AsRef<T>) + Sync { }
unsafe impl<T: ?Sized> Trait for dyn (::std::convert::AsRef<T>) + Send + Sync { }
unsafe impl Trait for dyn (::std::error::Error) + Send { }
unsafe impl Trait for dyn (::std::error::Error) + Sync { }
unsafe impl Trait for dyn (::std::error::Error) + Send + Sync { }
unsafe impl Trait for dyn (::std::fmt::Binary) + Send { }
unsafe impl Trait for dyn (::std::fmt::Binary) + Sync { }
unsafe impl Trait for dyn (::std::fmt::Binary) + Send + Sync { }
unsafe impl Trait for dyn (::std::fmt::Debug) + Send { }
unsafe impl Trait for dyn (::std::fmt::Debug) + Sync { }
unsafe impl Trait for dyn (::std::fmt::Debug) + Send + Sync { }
unsafe impl Trait for dyn (::std::fmt::Display) + Send { }
unsafe impl Trait for dyn (::std::fmt::Display) + Sync { }
unsafe impl Trait for dyn (::std::fmt::Display) + Send + Sync { }
unsafe impl Trait for dyn (::std::fmt::LowerExp) + Send { }
unsafe impl Trait for dyn (::std::fmt::LowerExp) + Sync { }
unsafe impl Trait for dyn (::std::fmt::LowerExp) + Send + Sync { }
unsafe impl Trait for dyn (::std::fmt::LowerHex) + Send { }
unsafe impl Trait for dyn (::std::fmt::LowerHex) + Sync { }
unsafe impl Trait for dyn (::std::fmt::LowerHex) + Send + Sync { }
unsafe impl Trait for dyn (::std::fmt::Octal) + Send { }
unsafe impl Trait for dyn (::std::fmt::Octal) + Sync { }
unsafe impl Trait for dyn (::std::fmt::Octal) + Send + Sync { }
unsafe impl Trait for dyn (::std::fmt::Pointer) + Send { }
unsafe impl Trait for dyn (::std::fmt::Pointer) + Sync { }
unsafe impl Trait for dyn (::std::fmt::Pointer) + Send + Sync { }
unsafe impl Trait for dyn (::std::fmt::UpperExp) + Send { }
unsafe impl Trait for dyn (::std::fmt::UpperExp) + Sync { }
unsafe impl Trait for dyn (::std::fmt::UpperExp) + Send + Sync { }
unsafe impl Trait for dyn (::std::fmt::UpperHex) + Send { }
unsafe impl Trait for dyn (::std::fmt::UpperHex) + Sync { }
unsafe impl Trait for dyn (::std::fmt::UpperHex) + Send + Sync { }
unsafe impl Trait for dyn (::std::fmt::Write) + Send { }
unsafe impl Trait for dyn (::std::fmt::Write) + Sync { }
unsafe impl Trait for dyn (::std::fmt::Write) + Send + Sync { }
unsafe impl Trait for dyn (::std::hash::Hasher) + Send { }
unsafe impl Trait for dyn (::std::hash::Hasher) + Sync { }
unsafe impl Trait for dyn (::std::hash::Hasher) + Send + Sync { }
unsafe impl Trait for dyn (::std::io::BufRead) + Send { }
unsafe impl Trait for dyn (::std::io::BufRead) + Sync { }
unsafe impl Trait for dyn (::std::io::BufRead) + Send + Sync { }
unsafe impl Trait for dyn (::std::io::Read) + Send { }
unsafe impl Trait for dyn (::std::io::Read) + Sync { }
unsafe impl Trait for dyn (::std::io::Read) + Send + Sync { }
unsafe impl Trait for dyn (::std::io::Seek) + Send { }
unsafe impl Trait for dyn (::std::io::Seek) + Sync { }
unsafe impl Trait for dyn (::std::io::Seek) + Send + Sync { }
unsafe impl Trait for dyn (::std::io::Write) + Send { }
unsafe impl Trait for dyn (::std::io::Write) + Sync { }
unsafe impl Trait for dyn (::std::io::Write) + Send + Sync { }
unsafe impl<T, I> Trait for dyn (::std::iter::IntoIterator<IntoIter=I, Item=T>) { }
unsafe impl<T> Trait for dyn (::std::iter::Iterator<Item=T>) + Send { }
unsafe impl<T> Trait for dyn (::std::iter::Iterator<Item=T>) + Sync { }
unsafe impl<T> Trait for dyn (::std::iter::Iterator<Item=T>) + Send + Sync { }
unsafe impl Trait for dyn (::std::marker::Send) + Send { }
unsafe impl Trait for dyn (::std::marker::Send) + Sync { }
unsafe impl Trait for dyn (::std::marker::Send) + Send + Sync { }
//~^ ERROR conflicting implementations of trait `Trait` for type
unsafe impl Trait for dyn (::std::marker::Sync) + Send { }
//~^ ERROR conflicting implementations of trait `Trait` for type
unsafe impl Trait for dyn (::std::marker::Sync) + Sync { }
unsafe impl Trait for dyn (::std::marker::Sync) + Send + Sync { }
//~^ ERROR conflicting implementations of trait `Trait` for type
unsafe impl Trait for dyn (::std::ops::Drop) + Send { }
unsafe impl Trait for dyn (::std::ops::Drop) + Sync { }
unsafe impl Trait for dyn (::std::ops::Drop) + Send + Sync { }
unsafe impl Trait for dyn (::std::string::ToString) + Send { }
unsafe impl Trait for dyn (::std::string::ToString) + Sync { }
unsafe impl Trait for dyn (::std::string::ToString) + Send + Sync { }
fn assert_trait<T: Trait + ?Sized>() {}

fn main() {
    assert_trait::<dyn Send>();
    assert_trait::<dyn Sync>();
    assert_trait::<dyn Send + Sync>();
}
