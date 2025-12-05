use std::fmt::Display;
use std::sync::Arc;

pub struct AnyId(());

impl PartialEq<Self> for AnyId {
    fn eq(&self, _: &Self) -> bool {
        todo!()
    }
}

impl<T: Identifier> PartialEq<T> for AnyId {
    fn eq(&self, _: &T) -> bool {
        todo!()
    }
}

impl<T: Identifier> From<T> for AnyId {
    fn from(_: T) -> Self {
        todo!()
    }
}

pub trait Identifier: Display + 'static {}

impl<T> Identifier for T where T: PartialEq + Display + 'static {}
