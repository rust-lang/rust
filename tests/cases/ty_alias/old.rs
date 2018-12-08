#![allow(type_alias_bounds)]

pub type A = u8;
pub type B<'a, T> = &'a T;
pub type C<'a, T> = &'a T;
pub type D<'a, T> = &'a T;
pub type E<'a, T> = &'a T;
pub type F<'a, U=u8> = &'a U;
pub type G<'a, T> = &'a T;
