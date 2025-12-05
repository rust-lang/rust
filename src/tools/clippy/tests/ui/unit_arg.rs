//@aux-build: proc_macros.rs
//@no-rustfix: overlapping suggestions
#![warn(clippy::unit_arg)]
#![allow(unused_must_use, unused_variables)]
#![allow(
    clippy::let_unit_value,
    clippy::needless_question_mark,
    clippy::never_loop,
    clippy::no_effect,
    clippy::or_fun_call,
    clippy::self_named_constructors,
    clippy::uninlined_format_args,
    clippy::unnecessary_wraps,
    clippy::unused_unit
)]

extern crate proc_macros;

use proc_macros::with_span;
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

fn baz<T: Debug>(t: T) {
    foo(t);
}

trait Tr {
    type Args;
    fn do_it(args: Self::Args);
}

struct A;
impl Tr for A {
    type Args = ();
    fn do_it(_: Self::Args) {}
}

struct B;
impl Tr for B {
    type Args = <A as Tr>::Args;

    fn do_it(args: Self::Args) {
        A::do_it(args)
    }
}

fn bad() {
    foo({
        //~^ unit_arg
        1;
    });
    foo(foo(1));
    //~^ unit_arg
    foo({
        //~^ unit_arg
        foo(1);
        foo(2);
    });
    let b = Bar;
    b.bar({
        //~^ unit_arg
        1;
    });
    taking_multiple_units(foo(0), foo(1));
    //~^ unit_arg
    taking_multiple_units(foo(0), {
        //~^ unit_arg
        foo(1);
        foo(2);
    });
    taking_multiple_units(
        //~^ unit_arg
        {
            foo(0);
            foo(1);
        },
        {
            foo(2);
            foo(3);
        },
    );
    // here Some(foo(2)) isn't the top level statement expression, wrap the suggestion in a block
    None.or(Some(foo(2)));
    //~^ unit_arg
    // in this case, the suggestion can be inlined, no need for a surrounding block
    // foo(()); foo(()) instead of { foo(()); foo(()) }
    foo(foo(()));
    //~^ unit_arg
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
    let named_unit_arg = ();
    foo(named_unit_arg);
    baz(());
    B::do_it(());
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
    //~^ unit_arg
}

fn taking_multiple_units(a: (), b: ()) {}

fn proc_macro() {
    with_span!(span taking_multiple_units(unsafe { (); }, 'x: loop { break 'x (); }));
}

fn main() {
    bad();
    ok();
}

fn issue14857() {
    let fn_take_unit = |_: ()| {};
    fn some_other_fn(_: &i32) {}

    macro_rules! mac {
        (def) => {
            Default::default()
        };
        (func $f:expr) => {
            $f()
        };
        (nonempty_block $e:expr) => {{
            some_other_fn(&$e);
            $e
        }};
    }
    fn_take_unit(mac!(def));
    //~^ unit_arg
    fn_take_unit(mac!(func Default::default));
    //~^ unit_arg
    fn_take_unit(mac!(nonempty_block Default::default()));
    //~^ unit_arg
}
