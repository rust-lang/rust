// Test for issue #153853: ICE when using paren sugar with inferred types
#![feature(unboxed_closures)]
#[rustc_paren_sugar]
trait Tr<'a, 'b, T> {
    fn method() {}
}

fn main() {
    <u8 as Tr(&u8)>::method;
}
