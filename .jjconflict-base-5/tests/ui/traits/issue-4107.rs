//@ run-pass
#![allow(dead_code)]

pub fn main() {
    let _id: &Mat2<f64> = &Matrix::identity(1.0);
}

pub trait Index<Index,Result> { fn get(&self, _: Index) -> Result { panic!() } }
pub trait Dimensional<T>: Index<usize, T> { }

pub struct Mat2<T> { x: T }
pub struct Vec2<T> { x: T }

impl<T> Dimensional<Vec2<T>> for Mat2<T> { }
impl<T> Index<usize, Vec2<T>> for Mat2<T> { }

impl<T> Dimensional<T> for Vec2<T> { }
impl<T> Index<usize, T> for Vec2<T> { }

pub trait Matrix<T,V>: Dimensional<V> {
    fn identity(t:T) -> Self;
}

impl<T> Matrix<T, Vec2<T>> for Mat2<T> {
    fn identity(t:T) -> Mat2<T> { Mat2{ x: t } }
}
