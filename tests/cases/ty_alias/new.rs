#![allow(type_alias_bounds)]

pub type A = u16;
pub type B<'a, 'b : 'a, T> = (&'a T, &'b T);
pub type C<T> = T;
pub type D<'a, T, U=Box<T>> = (&'a T, U);
pub type E<'a, T, U> = (&'a T, U);
pub type F<'a> = &'a u8;
pub type G<'a> = &'a u8;
