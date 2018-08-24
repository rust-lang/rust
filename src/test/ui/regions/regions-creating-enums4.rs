enum ast<'a> {
    num(usize),
    add(&'a ast<'a>, &'a ast<'a>)
}

fn mk_add_bad2<'a,'b>(x: &'a ast<'a>, y: &'a ast<'a>, z: &ast) -> ast<'b> {
    ast::add(x, y) //~ ERROR cannot infer
}

fn main() {
}
