// Check that `<-` and `in` syntax gets a hard error.

fn foo() {
    let (x, y) = (0, 0);
    x <- y; //~ ERROR unexpected token: `<-`
}

fn main() {
    let (foo, bar) = (0, 0);
    in(foo) { bar }; //~ ERROR expected expression, found keyword `in`
}
