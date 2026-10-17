//@ check-pass

use std::fmt::Debug;
use std::iter::Empty;
use std::ops::Range;

trait Lam<Binder> { type App; }

#[derive(Clone)]
struct L1;
impl<'a> Lam<&'a u8> for L1 { type App = u8; }

#[derive(Clone)]
struct L2;
impl<'a, 'b> Lam<&'a &'b u8> for L2 { type App = u8; }

trait Case1 {
    type A: Iterator<Item: Debug>;
    type B: Iterator<Item: 'static>;
}

pub struct S1;
impl Case1 for S1 {
    type A = Empty<String>;
    type B = Range<u16>;
}

// Ensure we don't have opaque `impl Trait` desugaring:

// What is this supposed to mean? Rustc currently lowers `: Default` in the
// bounds of `Out`, but trait selection can't find the bound since it applies
// to a type other than `Self::Out`.
pub trait Foo { type Out: Baz<Assoc: Default>; }
pub trait Baz { type Assoc; }

#[derive(Default)]
struct S2;
#[derive(Default)]
struct S3;
struct S4;
struct S5;
struct S6;
struct S7;

impl Foo for S6 { type Out = S4; }
impl Foo for S7 { type Out = S5; }

impl Baz for S4 { type Assoc = S2; }
impl Baz for S5 { type Assoc = S3; }

fn main() {}
