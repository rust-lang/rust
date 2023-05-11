// edition:2021

// check-pass
#![warn(unused)]
#![allow(dead_code)]

#[derive(Debug)]
struct MyStruct {
    a: i32,
    b: i32,
}

pub fn unintentional_copy_one() {
    let mut a = 1;
    let mut last = MyStruct{ a: 1, b: 1};
    let mut f = move |s| {
        // This will not trigger a warning for unused variable
        // as last.a will be treated as a Non-tracked place
        last.a = s;
        a = s;
        //~^ WARN value assigned to `a` is never read
        //~| WARN unused variable: `a`
    };
    f(2);
    f(3);
    f(4);
}

pub fn unintentional_copy_two() {
    let mut a = 1;
    let mut sum = MyStruct{ a: 1, b: 0};
    (1..10).for_each(move |x| {
        // This will not trigger a warning for unused variable
        // as sum.b will be treated as a Non-tracked place
        sum.b += x;
        a += x; //~ WARN unused variable: `a`
    });
}

fn main() {
    unintentional_copy_one();
    unintentional_copy_two();
}
