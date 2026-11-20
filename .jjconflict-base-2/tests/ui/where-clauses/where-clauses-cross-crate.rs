//@ run-pass
//@ aux-build:where_clauses_xc.rs

extern crate where_clauses_xc;

use where_clauses_xc::{Equal, equal};

fn main() {
    println!("{}", equal(&1, &2));
    println!("{}", equal(&1, &1));
    println!("{}", "hello".equal(&"hello"));
    println!("{}", "hello".equals::<isize,&str>(&1, &1, &"foo", &"bar"));
}
