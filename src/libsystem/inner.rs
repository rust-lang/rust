pub mod prelude {
    pub use super::{AsInner, AsInnerMut, IntoInner, FromInner};
}

/// A trait for viewing representations from std types
pub trait AsInner<Inner: ?Sized> {
    fn as_inner(&self) -> &Inner;
}

/// A trait for viewing representations from std types
pub trait AsInnerMut<Inner: ?Sized> {
    fn as_inner_mut(&mut self) -> &mut Inner;
}

/// A trait for extracting representations from std types
pub trait IntoInner<Inner> {
    fn into_inner(self) -> Inner;
}

/// A trait for creating std types from internal representations
pub trait FromInner<Inner> {
    fn from_inner(inner: Inner) -> Self;
}
