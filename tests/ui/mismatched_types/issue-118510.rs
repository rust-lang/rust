pub enum Sexpr<'a, S> {
    Ident(&'a mut S),
}

fn map<Foo, T, F: FnOnce(&Foo) -> T>(f: F) {}

fn main() {
    map(Sexpr::Ident);
    //~^ ERROR type mismatch in function arguments
}
