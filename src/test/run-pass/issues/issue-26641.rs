// run-pass
struct Parser<'a>(Box<FnMut(Parser) + 'a>);

fn main() {
    let _x = Parser(Box::new(|_|{}));
}
