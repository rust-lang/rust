pub enum Sexpr<'a> {
    Ident(&'a str),
}

fn map<'a, F: Fn(String) -> Sexpr<'a>>(f: F) {}

fn main() {
    map(Sexpr::Ident);
    //~^ ERROR type mismatch in function arguments
}
