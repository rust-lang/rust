// run-rustfix

#![allow(unused)]
#![deny(explicit_outlives_requirements)]

use std::fmt::{Debug, Display};


// Programmatically generated examples!
//
// Exercise outlives bounds for each of the following parameter/position
// combinations—
//
// • one generic parameter (T) bound inline
// • one parameter (T) with a where clause
// • two parameters (T and U), both bound inline
// • two parameters (T and U), one bound inline, one with a where clause
// • two parameters (T and U), both with where clauses
//
// —and for every permutation of 0, 1, or 2 lifetimes to outlive and 0 or 1
// trait bounds distributed among said parameters (subject to no where clause
// being empty and the struct having at least one lifetime).


struct TeeOutlivesAy<'a, T: 'a> {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a T
}

struct TeeOutlivesAyIsDebug<'a, T: 'a + Debug> {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a T
}

struct TeeIsDebugOutlivesAy<'a, T: Debug + 'a> {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a T
}

struct TeeOutlivesAyBee<'a, 'b, T: 'a + 'b> {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a &'b T
}

struct TeeOutlivesAyBeeIsDebug<'a, 'b, T: 'a + 'b + Debug> {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a &'b T
}

struct TeeIsDebugOutlivesAyBee<'a, 'b, T: Debug + 'a + 'b> {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a &'b T
}

struct TeeWhereOutlivesAy<'a, T> where T: 'a {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a T
}

struct TeeWhereOutlivesAyIsDebug<'a, T> where T: 'a + Debug {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a T
}

struct TeeWhereIsDebugOutlivesAy<'a, T> where T: Debug + 'a {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a T
}

struct TeeWhereOutlivesAyBee<'a, 'b, T> where T: 'a + 'b {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a &'b T
}

struct TeeWhereOutlivesAyBeeIsDebug<'a, 'b, T> where T: 'a + 'b + Debug {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a &'b T
}

struct TeeWhereIsDebugOutlivesAyBee<'a, 'b, T> where T: Debug + 'a + 'b {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a &'b T
}

struct TeeYooOutlivesAy<'a, T, U: 'a> {
    //~^ ERROR outlives requirements can be inferred
    tee: T,
    yoo: &'a U
}

struct TeeYooOutlivesAyIsDebug<'a, T, U: 'a + Debug> {
    //~^ ERROR outlives requirements can be inferred
    tee: T,
    yoo: &'a U
}

struct TeeYooIsDebugOutlivesAy<'a, T, U: Debug + 'a> {
    //~^ ERROR outlives requirements can be inferred
    tee: T,
    yoo: &'a U
}

struct TeeOutlivesAyYooIsDebug<'a, T: 'a, U: Debug> {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a T,
    yoo: U
}

struct TeeYooOutlivesAyBee<'a, 'b, T, U: 'a + 'b> {
    //~^ ERROR outlives requirements can be inferred
    tee: T,
    yoo: &'a &'b U
}

struct TeeYooOutlivesAyBeeIsDebug<'a, 'b, T, U: 'a + 'b + Debug> {
    //~^ ERROR outlives requirements can be inferred
    tee: T,
    yoo: &'a &'b U
}

struct TeeYooIsDebugOutlivesAyBee<'a, 'b, T, U: Debug + 'a + 'b> {
    //~^ ERROR outlives requirements can be inferred
    tee: T,
    yoo: &'a &'b U
}

struct TeeOutlivesAyBeeYooIsDebug<'a, 'b, T: 'a + 'b, U: Debug> {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a &'b T,
    yoo: U
}

struct TeeYooWhereOutlivesAy<'a, T, U> where U: 'a {
    //~^ ERROR outlives requirements can be inferred
    tee: T,
    yoo: &'a U
}

struct TeeYooWhereOutlivesAyIsDebug<'a, T, U> where U: 'a + Debug {
    //~^ ERROR outlives requirements can be inferred
    tee: T,
    yoo: &'a U
}

struct TeeYooWhereIsDebugOutlivesAy<'a, T, U> where U: Debug + 'a {
    //~^ ERROR outlives requirements can be inferred
    tee: T,
    yoo: &'a U
}

struct TeeOutlivesAyYooWhereIsDebug<'a, T: 'a, U> where U: Debug {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a T,
    yoo: U
}

struct TeeYooWhereOutlivesAyBee<'a, 'b, T, U> where U: 'a + 'b {
    //~^ ERROR outlives requirements can be inferred
    tee: T,
    yoo: &'a &'b U
}

struct TeeYooWhereOutlivesAyBeeIsDebug<'a, 'b, T, U> where U: 'a + 'b + Debug {
    //~^ ERROR outlives requirements can be inferred
    tee: T,
    yoo: &'a &'b U
}

struct TeeYooWhereIsDebugOutlivesAyBee<'a, 'b, T, U> where U: Debug + 'a + 'b {
    //~^ ERROR outlives requirements can be inferred
    tee: T,
    yoo: &'a &'b U
}

struct TeeOutlivesAyBeeYooWhereIsDebug<'a, 'b, T: 'a + 'b, U> where U: Debug {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a &'b T,
    yoo: U
}

struct TeeWhereOutlivesAyYooWhereIsDebug<'a, T, U> where T: 'a, U: Debug {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a T,
    yoo: U
}

struct TeeWhereOutlivesAyBeeYooWhereIsDebug<'a, 'b, T, U> where T: 'a + 'b, U: Debug {
    //~^ ERROR outlives requirements can be inferred
    tee: &'a &'b T,
    yoo: U
}


// But outlives inference for 'static lifetimes is under a separate
// feature-gate for now
// (https://github.com/rust-lang/rust/issues/44493#issuecomment-407846046).
struct StaticRef<T: 'static> {
    field: &'static T
}


fn main() {}
