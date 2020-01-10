enum Token { LeftParen, RightParen, Plus, Minus, /* etc */ }
struct Struct {
    a: usize,
}

fn use_token(token: &Token) { unimplemented!() }

fn main() {
    use_token(&Token::Homura); //~ ERROR no variant or associated item named `Homura`
    Struct::method(); //~ ERROR no function or associated item named `method` found
    Struct::method; //~ ERROR no function or associated item named `method` found
    Struct::Assoc; //~ ERROR no associated item named `Assoc` found
}
