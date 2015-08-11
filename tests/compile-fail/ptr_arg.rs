#![feature(plugin)]
#![plugin(clippy)]

#[deny(ptr_arg)]
#[allow(unused)]
fn do_vec(x: &Vec<i64>) { //~ERROR: Writing '&Vec<_>' instead of '&[_]'
    //Nothing here
}

#[deny(ptr_arg)]
#[allow(unused)]
fn do_str(x: &String) { //~ERROR
    //Nothing here either
}

fn main() {
    let x = vec![1i64, 2, 3];
    do_vec(&x);
    do_str(&"hello".to_owned());
}
