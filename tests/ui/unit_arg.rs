#![warn(clippy::unit_arg)]
#![allow(clippy::no_effect, unused_must_use, unused_variables)]

use std::fmt::Debug;

fn foo<T: Debug>(t: T) {
    println!("{:?}", t);
}

fn foo3<T1: Debug, T2: Debug, T3: Debug>(t1: T1, t2: T2, t3: T3) {
    println!("{:?}, {:?}, {:?}", t1, t2, t3);
}

struct Bar;

impl Bar {
    fn bar<T: Debug>(&self, t: T) {
        println!("{:?}", t);
    }
}

fn bad() {
    foo({
        1;
    });
    foo(foo(1));
    foo({
        foo(1);
        foo(2);
    });
    let b = Bar;
    b.bar({
        1;
    });
    taking_multiple_units(foo(0), foo(1));
    taking_multiple_units(foo(0), {
        foo(1);
        foo(2);
    });
    taking_multiple_units(
        {
            foo(0);
            foo(1);
        },
        {
            foo(2);
            foo(3);
        },
    );
}

fn ok() {
    foo(());
    foo(1);
    foo({ 1 });
    foo3("a", 3, vec![3]);
    let b = Bar;
    b.bar({ 1 });
    b.bar(());
    question_mark();
}

fn question_mark() -> Result<(), ()> {
    Ok(Ok(())?)?;
    Ok(Ok(()))??;
    Ok(())
}

#[allow(dead_code)]
mod issue_2945 {
    fn unit_fn() -> Result<(), i32> {
        Ok(())
    }

    fn fallible() -> Result<(), i32> {
        Ok(unit_fn()?)
    }
}

#[allow(dead_code)]
fn returning_expr() -> Option<()> {
    Some(foo(1))
}

fn taking_multiple_units(a: (), b: ()) {}

fn main() {
    bad();
    ok();
}
