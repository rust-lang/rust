//@ compile-flags: -Zunstable-options --generate-link-to-definition
//@ aux-build:source_code.rs
//@ build-aux-docs

#![feature(rustc_attrs)]

#![crate_name = "foo"]

extern crate source_code;

//@ has 'src/foo/check-source-code-urls-to-def.rs.html'

//@ has - '//pre[@class="rust"]//a[@href="auxiliary/source-code-bar.rs.html#1-17"]' 'bar'
#[path = "auxiliary/source-code-bar.rs"]
pub mod bar;

//@ count - '//pre[@class="rust"]//a[@href="auxiliary/source-code-bar.rs.html#5-7"]' 4
use bar::Bar;
//@ has - '//pre[@class="rust"]//a[@href="auxiliary/source-code-bar.rs.html#13-17"]' 'self'
//@ has - '//pre[@class="rust"]//a[@href="auxiliary/source-code-bar.rs.html#14-16"]' 'Trait'
use bar::sub::{self, Trait};

pub struct Foo;

impl Foo {
    fn hello(&self) {}
}

fn babar() {}

//@ has - '//pre[@class="rust"]//a/@href' '/struct.String.html'
//@ has - '//pre[@class="rust"]//a/@href' '/primitive.u32.html'
//@ has - '//pre[@class="rust"]//a/@href' '/primitive.str.html'
// The 5 links to line 23 and the line 23 itself.
//@ count - '//pre[@class="rust"]//a[@href="#23"]' 6
//@ has - '//pre[@class="rust"]//a[@href="../../source_code/struct.SourceCode.html"]' \
//        'SourceCode'
pub fn foo(a: u32, b: &str, c: String, d: Foo, e: bar::Bar, f: source_code::SourceCode) {
    let x = 12;
    let y: Foo = Foo;
    let z: Bar = bar::Bar { field: Foo };
    babar();
    //@ has - '//pre[@class="rust"]//a[@href="#26"]' 'hello'
    y.hello();
}

//@ has - '//pre[@class="rust"]//a[@href="auxiliary/source-code-bar.rs.html#14-16"]' 'bar::sub::Trait'
//@ has - '//pre[@class="rust"]//a[@href="auxiliary/source-code-bar.rs.html#14-16"]' 'Trait'
pub fn foo2<T: bar::sub::Trait, V: Trait>(t: &T, v: &V, b: bool) {}

pub trait AnotherTrait {}
pub trait WhyNot {}

//@ has - '//pre[@class="rust"]//a[@href="#51"]' 'AnotherTrait'
//@ has - '//pre[@class="rust"]//a[@href="#52"]' 'WhyNot'
pub fn foo3<T, V>(t: &T, v: &V)
where
    T: AnotherTrait,
    V: WhyNot
{}

pub trait AnotherTrait2 {}

//@ has - '//pre[@class="rust"]//a[@href="#62"]' 'AnotherTrait2'
pub fn foo4() {
    let x: Vec<&dyn AnotherTrait2> = Vec::new();
}

//@ has - '//pre[@class="rust"]//a[@href="../../foo/primitive.bool.html"]' 'bool'
#[rustc_doc_primitive = "bool"]
mod whatever {}
