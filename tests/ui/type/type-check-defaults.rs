use std::iter::FromIterator;
use std::vec::IntoIter;
use std::ops::Add;

struct Foo<T, U: FromIterator<T>>(T, U);
struct WellFormed<Z = Foo<i32, i32>>(Z);
//~^ ERROR a value of type `i32` cannot be built from an iterator over elements of type `i32`
struct WellFormedNoBounds<Z:?Sized = Foo<i32, i32>>(Z);
//~^ ERROR a value of type `i32` cannot be built from an iterator over elements of type `i32`

struct Bounds<T:Copy=String>(T);
//~^ ERROR the trait bound `String: Copy` is not satisfied [E0277]

struct WhereClause<T=String>(T) where T: Copy;
//~^ ERROR the trait bound `String: Copy` is not satisfied [E0277]

trait TraitBound<T:Copy=String> {}
//~^ ERROR the trait bound `String: Copy` is not satisfied [E0277]

trait Super<T: Copy> { }
trait Base<T = String>: Super<T> { }
//~^ ERROR the trait bound `T: Copy` is not satisfied [E0277]

trait ProjectionPred<T:Iterator = IntoIter<i32>> where T::Item : Add<u8> {}
//~^ ERROR cannot add `u8` to `i32` [E0277]

fn main() { }
