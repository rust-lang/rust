// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    let _id: &Mat2<float> = &Matrix::identity();
}

pub trait Index<Index,Result> { }
pub trait Dimensional<T>: Index<uint, T> { }

pub struct Mat2<T> { x: () }
pub struct Vec2<T> { x: () }

impl<T> Dimensional<Vec2<T>> for Mat2<T> { }
impl<T> Index<uint, Vec2<T>> for Mat2<T> { }

impl<T> Dimensional<T> for Vec2<T> { }
impl<T> Index<uint, T> for Vec2<T> { }

pub trait Matrix<T,V>: Dimensional<V> {
    fn identity() -> Self;
}

impl<T> Matrix<T, Vec2<T>> for Mat2<T> {
    fn identity() -> Mat2<T> { Mat2{ x: () } }
}
