#![feature(plugin)]
#![plugin(clippy)]
#![allow(unused, dead_code, needless_lifetimes)]
#![deny(unused_lifetimes)]

fn empty() {

}


fn used_lt<'a>(x: &'a u8) {

}


fn unused_lt<'a>(x: u8) { //~ ERROR this lifetime

}

fn unused_lt_transitive<'a, 'b: 'a>(x: &'b u8) { //~ ERROR this lifetime
    // 'a is useless here since it's not directly bound
}

fn lt_return<'a, 'b: 'a>(x: &'b u8) -> &'a u8 {
    panic!()
}

fn lt_return_only<'a>() -> &'a u8 {
    panic!()
}

fn unused_lt_blergh<'a>(x: Option<Box<Send+'a>>) {

}


trait Foo<'a> {
    fn x(&self, a: &'a u8);
}

impl<'a> Foo<'a> for u8 {
    fn x(&self, a: &'a u8) {

    }
}
fn main() {

}
