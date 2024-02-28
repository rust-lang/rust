use std::iter::FromIterator;
use std::vec::IntoIter;
use std::ops::Add;

struct Foo<T, U: FromIterator<T>>(T, U);
struct WellFormed<Z = Foo<i32, i32>>(Z);
//~^ ERROR a value of type `i32` cannot be built from an iterator over elements of type `i32`
struct WellFormedNoBounds<Z:?Sized = Foo<i32, i32>>(Z);
//~^ ERROR a value of type `i32` cannot be built from an iterator over elements of type `i32`

struct Bounds<T:Copy=String>(T);
//~^ ERROR trait `Copy` is not implemented for `String`

struct WhereClause<T=String>(T) where T: Copy;
//~^ ERROR trait `Copy` is not implemented for `String`

trait TraitBound<T:Copy=String> {}
//~^ ERROR trait `Copy` is not implemented for `String`

trait Super<T: Copy> { }
trait Base<T = String>: Super<T> { }
//~^ ERROR trait `Copy` is not implemented for `T`

trait ProjectionPred<T:Iterator = IntoIter<i32>> where T::Item : Add<u8> {}
//~^ ERROR cannot add `u8` to `i32` [E0277]

fn main() { }
