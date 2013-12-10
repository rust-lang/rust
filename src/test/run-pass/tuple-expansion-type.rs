// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(macro_rules)];

fn same_type<T>(_: Option<T>, _: Option<T>) {}

macro_rules! ty_eq(
    ($T:ty, $U:ty) => (same_type(None::<$T>, None::<$U>))
)

fn main() {
    type A = (..(u8, i8));
    ty_eq!(A, (u8, i8));

    type B = (..((u8, i8)));
    ty_eq!(B, (u8, i8));

    type C = fn(..(u8, i8)) -> int;
    ty_eq!(C, fn(u8, i8) -> int);

    type D = 'static|..(u8, i8)| -> int;
    ty_eq!(D, 'static|u8, i8| -> int);

    type E = proc(..(u8, i8)) -> int;
    ty_eq!(E, proc(u8, i8) -> int);

    type F<T, U> = (T, U, T);
    ty_eq!(F<char, int>, (char, int, char));

    type G = (..F<..(u8, i8)>);
    ty_eq!(G, (u8, i8, u8));

    type TupleWrap<T, U> = (T, ..U, T);
    ty_eq!(TupleWrap<char, (u8, i8)>, (char, u8, i8, char));

    type I<T, U> = (T, ..F<T, U>, T);
    ty_eq!(I<char, (u8, i8)>, (char, char, (u8, i8), char, char));

    type TupleConcat<T, U> = (..T, ..U);
    ty_eq!(TupleConcat<(char, int), (u8, i8)>, (char, int, u8, i8));
}
