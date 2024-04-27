//@ run-pass
struct Parser<'a>(#[allow(dead_code)] Box<dyn FnMut(Parser) + 'a>);

fn main() {
    let _x = Parser(Box::new(|_|{}));
}
