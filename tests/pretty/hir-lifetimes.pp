//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-lifetimes.pp

// This tests the pretty-printing of lifetimes in lots of ways.

#![allow(unused)]
#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;

struct Foo<'a> {
    x: &'a u32,
}

impl <'a> Foo<'a> {
    fn f<'b>(x: &'b u32) { }
}

impl  Foo<'_> {
    fn a(x: &'_ u32) { }

    fn b(x: &'_ u32) { }

    fn c(x: &'_ u32, y: &'static u32) { }

    // FIXME: `'a` before `self` is omitted
    fn d<'a>(&self, x: &'a u32) { }

    // FIXME: impl Traits printed as just `/*impl Trait*/`, ugh
    fn iter1<'a>(&self)
        -> /*impl Trait*/ { #[lang = "Range"] { start: 0, end: 1 } }

    fn iter2(&self)
        -> /*impl Trait*/ { #[lang = "Range"] { start: 0, end: 1 } }
}

fn a(x: Foo<'_>) { }

fn b<'a>(x: Foo<'a>) { }

struct Bar<'a, 'b, 'c, T> {
    x: &'a u32,
    y: &'b &'c u32,
    z: T,
}

fn f1<'a, 'b, T>(x: Bar<'a, 'b, '_, T>) { }

fn f2(x: Bar<'_, '_, '_, u32>) { }

trait MyTrait<'a, 'b> {
    fn f(&self, x: Foo<'a>, y: Foo<'b>);
}

impl <'a, 'b, 'c, T> MyTrait<'a, 'b> for Bar<'a, 'b, 'c, T> {
    fn f(&self, x: Foo<'a>, y: Foo<'b>) { }
}

fn g(x: &'_ dyn for<'a, 'b> MyTrait<'a, 'b>) { }

trait Blah { }

type T<'a> = dyn Blah + 'a;

type Q<'a> = dyn MyTrait<'a, 'a> + 'a;

fn h<'b, F>(f: F, y: Foo<'b>) where F: for<'d> MyTrait<'d, 'b> { }

// FIXME(?): attr printing is weird
#[attr = Repr([ReprC])]
struct S<'a>(&'a u32);

extern "C" {
    unsafe fn g1(s: S<'_>);
    unsafe fn g2(s: S<'_>);
    unsafe fn g3<'a>(s: S<'a>);
}

struct St<'a> {
    x: &'a u32,
}

fn f() { { let _ = St { x: &0 }; }; { let _ = St { x: &0 }; }; }

struct Name<'a>(&'a str);

const A: Name<'_> = Name("a");
const B: &'_ str = "";
static C: &'_ str = "";
static D: &'static str = "";

fn tr(_: Box<dyn Blah>) { }

fn main() { }
