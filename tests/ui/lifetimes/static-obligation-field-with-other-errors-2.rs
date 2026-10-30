// Regression test for #165052
// This used to ICE
//@ check-fail

enum Ast<'a> {
    Add(&'a Ast<'static>, &'a Ast<'a>),
}
fn mk_add_bad1<'a, 'b>(x: &'a Ast, y: &'b Ast) -> Ast<'a> {
    Ast::Add(x, y)
    //~^ ERROR explicit lifetime required in the type of `y`
    //~| ERROR lifetime may not live long enough
    //~| ERROR lifetime may not live long enough
}

fn main() {}
