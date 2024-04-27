fn sum<I: Iterator<Item = ()>>(i: I) -> i32 where I::Item = i32 {
//~^ ERROR equality constraints are not yet supported in `where` clauses
    panic!()
}
fn sum2<I: Iterator>(i: I) -> i32 where I::Item = i32 {
//~^ ERROR equality constraints are not yet supported in `where` clauses
    panic!()
}
fn sum3<J: Iterator>(i: J) -> i32 where I::Item = i32 {
//~^ ERROR equality constraints are not yet supported in `where` clauses
//~| ERROR failed to resolve: use of undeclared type `I`
    panic!()
}

use std::iter::FromIterator;

struct X {}

impl FromIterator<bool> for X {
    fn from_iter<T>(_: T) -> Self where T: IntoIterator, IntoIterator::Item = A,
        //~^ ERROR equality constraints are not yet supported in `where` clauses
        //~| ERROR cannot find type `A` in this scope
    {
        todo!()
    }
}

struct Y {}

impl FromIterator<bool> for Y {
    fn from_iter<T>(_: T) -> Self where T: IntoIterator, T::Item = A,
        //~^ ERROR equality constraints are not yet supported in `where` clauses
        //~| ERROR cannot find type `A` in this scope
    {
        todo!()
    }
}

struct Z {}

impl FromIterator<bool> for Z {
    fn from_iter<T: IntoIterator>(_: T) -> Self where IntoIterator::Item = A,
        //~^ ERROR equality constraints are not yet supported in `where` clauses
        //~| ERROR cannot find type `A` in this scope
    {
        todo!()
    }
}

struct K {}

impl FromIterator<bool> for K {
    fn from_iter<T: IntoIterator>(_: T) -> Self where T::Item = A,
        //~^ ERROR equality constraints are not yet supported in `where` clauses
        //~| ERROR cannot find type `A` in this scope
    {
        todo!()
    }
}

struct L {}

impl FromIterator<bool> for L {
    fn from_iter<T>(_: T) -> Self where IntoIterator::Item = A, T: IntoIterator,
        //~^ ERROR equality constraints are not yet supported in `where` clauses
        //~| ERROR cannot find type `A` in this scope
    {
        todo!()
    }
}

struct M {}

impl FromIterator<bool> for M {
    fn from_iter<T>(_: T) -> Self where T::Item = A, T: IntoIterator,
        //~^ ERROR equality constraints are not yet supported in `where` clauses
        //~| ERROR cannot find type `A` in this scope
    {
        todo!()
    }
}
fn main() {}
