//@ run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]


enum ast<'a> {
    num(usize),
    add(&'a ast<'a>, &'a ast<'a>)
}

fn mk_add_ok<'a>(x: &'a ast<'a>, y: &'a ast<'a>, _z: &ast) -> ast<'a> {
    ast::add(x, y)
}

pub fn main() {
}
