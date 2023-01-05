// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

// pretty-expanded FIXME #23616

enum ast<'a> {
    num(usize),
    add(&'a ast<'a>, &'a ast<'a>)
}

fn mk_add_ok<'r>(x: &'r ast<'r>, y: &'r ast<'r>) -> ast<'r> {
    ast::add(x, y)
}

pub fn main() {
}
