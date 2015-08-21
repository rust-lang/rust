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
