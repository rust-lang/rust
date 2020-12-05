// run-pass
struct Parser<'a>(Box<dyn FnMut(Parser) + 'a>);

fn main() {
    let _x = Parser(Box::new(|_|{}));
}
