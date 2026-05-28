//! regression test for issue <https://github.com/rust-lang/rust/issues/23173>
enum Token {
    LeftParen,
    RightParen,
    Plus,
    Minus, /* etc */
}
struct Struct {
    a: usize,
}

fn use_token(token: &Token) {
    unimplemented!()
}

fn main() {
    use_token(&Token::Homura); //~ ERROR no variant, associated function, or constant named `Homura`
    Struct::method(); //~ ERROR no associated function or constant named `method` found
    Struct::method; //~ ERROR no associated function or constant named `method` found
    Struct::Assoc; //~ ERROR no associated function or constant named `Assoc` found
}
