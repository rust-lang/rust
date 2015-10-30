#![feature(plugin)]
#![plugin(clippy)]
#![allow(unused)]
#![deny(ptr_arg)]

fn do_vec(x: &Vec<i64>) { //~ERROR writing `&Vec<_>` instead of `&[_]`
    //Nothing here
}

fn do_vec_mut(x: &mut Vec<i64>) { // no error here
    //Nothing here
}

fn do_str(x: &String) { //~ERROR writing `&String` instead of `&str`
    //Nothing here either
}

fn do_str_mut(x: &mut String) { // no error here
    //Nothing here either
}

fn main() {
}

trait Foo {
    type Item;
    fn do_vec(x: &Vec<i64>); //~ERROR writing `&Vec<_>`
    fn do_item(x: &Self::Item);
}

struct Bar;

// no error, in trait impl (#425)
impl Foo for Bar {
    type Item = Vec<u8>;
    fn do_vec(x: &Vec<i64>) {}
    fn do_item(x: &Vec<u8>) {}  
}
