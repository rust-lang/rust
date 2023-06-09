enum Ast<'a> {
    Num(usize),
    Add(&'a Ast<'a>, &'a Ast<'a>)
}

fn mk_add_bad2<'a,'b>(x: &'a Ast<'a>, y: &'a Ast<'a>, z: &Ast) -> Ast<'b> {
    Ast::Add(x, y)
    //~^ ERROR lifetime may not live long enough
}

fn main() {
}
