//@ edition:2021
//@ run-pass

#![allow(unused)]
#![allow(dead_code)]
#![allow(dropping_references)]

struct Int(i32);
struct B<'a>(&'a i32);

const I : Int = Int(0);
const REF_I : &'static Int = &I;

struct MyStruct<'a> {
   a: &'static Int,
   b: B<'a>,
}

fn foo<'a, 'b>(m: &'a MyStruct<'b>) -> impl FnMut() + 'static {
    let c = || drop(&m.a.0);
    c
}

fn main() {
    let t = 0;
    let s = MyStruct { a: REF_I, b: B(&t) };
    let _ = foo(&s);
}
