//! Regression test for https://github.com/rust-lang/rust/issues/26641
//@ run-pass
struct Parser<'a>(#[allow(dead_code)] Box<dyn FnMut(Parser) + 'a>);

fn main() {
    let _x = Parser(Box::new(|_|{}));
}
