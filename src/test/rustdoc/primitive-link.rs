#![crate_name = "foo"]

// ignore-tidy-linelength

// @has foo/struct.Foo.html '//*[@class="docblock"]/p/a[@href="https://doc.rust-lang.org/nightly/std/primitive.u32.html"]' 'u32'
// @has foo/struct.Foo.html '//*[@class="docblock"]/p/a[@href="https://doc.rust-lang.org/nightly/std/primitive.i64.html"]' 'i64'

/// It contains [`u32`] and [i64].
pub struct Foo;
