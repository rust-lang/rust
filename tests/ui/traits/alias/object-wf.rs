//@ check-pass

// This test checks that trait objects involving trait aliases are well-formed.

#![feature(trait_alias)]

trait Obj {}

trait _0 = Send + Sync;

// Just auto traits:

trait _1 = _0 + Send + Sync;

use std::marker::Unpin;

fn _f0() {
    let _: Box<dyn _0>;
    let _: Box<dyn _1>;
    let _: Box<dyn Unpin + _1 + Send + Sync>;
}

// Include dyn-compatible traits:

fn _f1() {
    let _: Box<dyn Obj + _0>;
    let _: Box<dyn Obj + _1>;
    let _: Box<dyn Obj + _1 + _0>;
}

// And when the dyn-compatible trait is in a trait alias:

trait _2 = Obj;

fn _f2() {
    let _: Box<dyn _2 + _0>;
    let _: Box<dyn _2 + _1>;
    let _: Box<dyn _2 + _1 + _0>;
}

// And it should also work when that trait is has auto traits to the right of it.

trait _3 = Obj + Unpin;

fn _f3() {
    let _: Box<dyn _3 + _0>;
    let _: Box<dyn _3 + _1>;
    let _: Box<dyn _3 + _1 + _0>;
}

// Nest the trait deeply:

trait _4 = _3;
trait _5 = _4 + Sync + _0 + Send;
trait _6 = _5 + Send + _1 + Sync;

fn _f4() {
    let _: Box<dyn _6 + _0>;
    let _: Box<dyn _6 + _1>;
    let _: Box<dyn _6 + _1 + _0>;
}

// Just nest the trait alone:

trait _7 = _2;
trait _8 = _7;
trait _9 = _8;

fn _f5() {
    let _: Box<dyn _9>;
}

// First bound is auto trait:

trait _10 = Send + Obj;
trait _11 = Obj + Send;
trait _12 = Sync + _11;
trait _13 = Send + _12;

fn f6() {
    let _: Box<dyn _10>;
    let _: Box<dyn _13>;
}

fn main() {}
