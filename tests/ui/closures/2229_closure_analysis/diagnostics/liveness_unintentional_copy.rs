//@ edition:2021

//@ check-pass
#![warn(unused)]
#![allow(dead_code)]

#[derive(Debug)]
struct MyStruct {
    a: i32,
    b: i32,
}

pub fn unintentional_copy_one() {
    let mut a = 1;
    //~^ WARN unused variable: `a`
    let mut last = MyStruct { a: 1, b: 1 };
    //~^ WARN unused variable: `last`
    let mut f = move |s| {
        last.a = s;
        //~^ WARN value captured by `last.a` is never read
        //~| WARN value assigned to `last.a` is never read
        a = s;
        //~^ WARN value captured by `a` is never read
        //~| WARN value assigned to `a` is never read
    };
    f(2);
    f(3);
    f(4);
}

pub fn unintentional_copy_two() {
    let mut a = 1;
    //~^ WARN unused variable: `a`
    let mut sum = MyStruct { a: 1, b: 0 };
    //~^ WARN unused variable: `sum`
    (1..10).for_each(move |x| {
        sum.b += x;
        //~^ WARN value captured by `sum.b` is never read
        //~| WARN value assigned to `sum.b` is never read
        a += x;
        //~^ WARN value captured by `a` is never read
        //~| WARN value assigned to `a` is never read
    });
}

fn main() {
    unintentional_copy_one();
    unintentional_copy_two();
}
