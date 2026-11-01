//@ check-pass
#![feature(type_alias_impl_trait)]

pub trait ValidTrait {}
type ImplTrait = impl ValidTrait;

/// This returns impl trait, but using a type alias
pub fn h() -> ImplTrait {
    (|| error::_in::impl_trait::alias::nested::closure())()
}
