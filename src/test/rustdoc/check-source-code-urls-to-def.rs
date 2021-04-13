// compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

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

// @has - '//a[@href="https://doc.rust-lang.org/nightly/alloc/string/struct.String.html"]' 'String'
// @count - '//a[@href="../../src/foo/check-source-code-urls-to-def.rs.html#17"]' 5
pub fn foo(a: u32, b: &str, c: String, d: Foo, e: bar::Bar) {
    let x = 12;
    let y: Foo = Foo;
    let z: Bar = bar::Bar { field: Foo };
    babar();
    // @has - '//a[@href="../../src/foo/check-source-code-urls-to-def.rs.html#20"]' 'hello'
    y.hello();
}

// @has - '//a[@href="../../src/foo/auxiliary/source-code-bar.rs.html#14-16"]' 'bar::sub::Trait'
// @has - '//a[@href="../../src/foo/auxiliary/source-code-bar.rs.html#14-16"]' 'Trait'
pub fn foo2<T: bar::sub::Trait, V: Trait>(t: &T, v: &V) {
}
