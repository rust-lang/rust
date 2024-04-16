//@ known-bug: #122552
//@ edition:2021

trait X {
    fn line_stream<'a, Repr>() -> Self::LineStreamFut<{ async {} }, Repr>;
}

struct Y;

pub fn main() {}
