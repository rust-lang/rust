//! <https://github.com/rust-lang/rust-clippy/issues/9746#issuecomment-1297132880>

trait Trait {}

struct Struct<'a> {
    _inner: &'a Struct<'a>,
}

impl Trait for Struct<'_> {}

fn example<'a>(s: &'a Struct) -> Box<Box<dyn Trait + 'a>> {
    Box::new(Box::new(Struct { _inner: s }))
}

fn main() {}
