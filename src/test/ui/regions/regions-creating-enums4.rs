// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

enum Ast<'a> {
    Num(usize),
    Add(&'a Ast<'a>, &'a Ast<'a>)
}

fn mk_add_bad2<'a,'b>(x: &'a Ast<'a>, y: &'a Ast<'a>, z: &Ast) -> Ast<'b> {
    Ast::Add(x, y)
    //[base]~^ ERROR cannot infer
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() {
}
