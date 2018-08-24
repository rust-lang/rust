enum ast<'a> {
    num(usize),
    add(&'a ast<'a>, &'a ast<'a>)
}

fn mk_add_bad1<'a,'b>(x: &'a ast<'a>, y: &'b ast<'b>) -> ast<'a> {
    ast::add(x, y) //~ ERROR 17:5: 17:19: lifetime mismatch [E0623]
}

fn main() {
}
