// run-pass

// This test checks that trait objects involving trait aliases are well-formed.

#![feature(trait_alias)]

trait Obj {}

trait _0 = Send + Sync;

// Just auto traits:

trait _1 = _0 + Send + Sync;

use std::marker::Unpin;

type _T01 = dyn _0;
type _T02 = dyn _1;
type _T03 = dyn Unpin + _1 + Send + Sync;

// Include object safe traits:

type _T10 = dyn Obj + _0;
type _T11 = dyn Obj + _1;
type _T12 = dyn Obj + _1 + _0;

// And when the object safe trait is in a trait alias:

trait _2 = Obj;

type _T20 = dyn _2 + _0;
type _T21 = dyn _2 + _1;
type _T22 = dyn _2 + _1 + _0;

// And it should also work when that trait is has auto traits to the right of it.

trait _3 = Obj + Unpin;

type _T30 = dyn _3 + _0;
type _T31 = dyn _3 + _1;
type _T32 = dyn _3 + _1 + _0;

// Nest the trait deeply:

trait _4 = _3;
trait _5 = _4 + Sync + _0 + Send;
trait _6 = _5 + Send + _1 + Sync;

type _T60 = dyn _6 + _0;
type _T61 = dyn _6 + _1;
type _T62 = dyn _6 + _1 + _0;

// Just nest the trait alone:

trait _7 = _2;
trait _8 = _7;
trait _9 = _8;

type _T9 = dyn _9;

// First bound is auto trait:

trait _10 = Send + Obj;
trait _11 = Obj + Send;
trait _12 = Sync + _11;
trait _13 = Send + _12;

type _T70 = dyn _0;
type _T71 = dyn _3;

fn main() {}
