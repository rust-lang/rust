// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
#[derive(Debug)]
enum Foo<'s> {
    V(&'s str)
}

fn f(arr: &[&Foo]) {
    for &f in arr {
        println!("{:?}", f);
    }
}

fn main() {}
