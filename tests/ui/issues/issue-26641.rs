// run-pass
struct Parser<'a>(#[allow(unused_tuple_struct_fields)] Box<dyn FnMut(Parser) + 'a>);

fn main() {
    let _x = Parser(Box::new(|_|{}));
}
