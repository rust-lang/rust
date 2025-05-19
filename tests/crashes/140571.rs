//@ known-bug: #140571
pub trait IsVoid {
    const IS_VOID: bool;
}
impl<T> IsVoid for T {
    default const IS_VOID: bool = false;
}
impl<T> Maybe<T> for () where T: NotVoid + ?Sized {}

pub trait NotVoid {}
impl<T> NotVoid for T where T: IsVoid<IS_VOID = false> + ?Sized {}

pub trait Maybe<T> {}
impl<T> Maybe<T> for T {}
