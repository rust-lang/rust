// compile-flags: -Zunstable-options --generate-link-to-definition
// aux-build:source_code.rs
// build-aux-docs

#![crate_name = "foo"]

extern crate source_code;

// @has 'src/foo/check-source-code-urls-to-def.rs.html'

// @has - '//a[@href="../../src/foo/auxiliary/source-code-bar.rs.html#1-17"]' 'bar'
#[path = "auxiliary/source-code-bar.rs"]
pub mod bar;

// @count - '//a[@href="../../src/foo/auxiliary/source-code-bar.rs.html#5-7"]' 4
use bar::Bar;
// @has - '//a[@href="../../src/foo/auxiliary/source-code-bar.rs.html#13-17"]' 'self'
// @has - '//a[@href="../../src/foo/auxiliary/source-code-bar.rs.html#14-16"]' 'Trait'
use bar::sub::{self, Trait};

pub struct Foo;

impl Foo {
    fn hello(&self) {}
}

fn babar() {}

// @has - '//a/@href' '/struct.String.html'
// @has - '//a/@href' '/primitive.u32.html'
// @has - '//a/@href' '/primitive.str.html'
// @count - '//a[@href="../../src/foo/check-source-code-urls-to-def.rs.html#21"]' 5
// @has - '//a[@href="../../source_code/struct.SourceCode.html"]' 'source_code::SourceCode'
pub fn foo(a: u32, b: &str, c: String, d: Foo, e: bar::Bar, f: source_code::SourceCode) {
    let x = 12;
    let y: Foo = Foo;
    let z: Bar = bar::Bar { field: Foo };
    babar();
    // @has - '//a[@href="../../src/foo/check-source-code-urls-to-def.rs.html#24"]' 'hello'
    y.hello();
}

// @has - '//a[@href="../../src/foo/auxiliary/source-code-bar.rs.html#14-16"]' 'bar::sub::Trait'
// @has - '//a[@href="../../src/foo/auxiliary/source-code-bar.rs.html#14-16"]' 'Trait'
pub fn foo2<T: bar::sub::Trait, V: Trait>(t: &T, v: &V, b: bool) {
}

// @has - '//a[@href="../../foo/primitive.bool.html"]' 'bool'
#[doc(primitive = "bool")]
mod whatever {}
