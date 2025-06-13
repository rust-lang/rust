#![warn(clippy::unit_arg)]
#![allow(unused_must_use, unused_variables)]
#![allow(clippy::no_effect, clippy::uninlined_format_args)]

use std::fmt::Debug;

fn foo<T: Debug>(t: T) {
    println!("{:?}", t);
}

fn foo3<T1: Debug, T2: Debug, T3: Debug>(t1: T1, t2: T2, t3: T3) {
    println!("{:?}, {:?}, {:?}", t1, t2, t3);
}

fn bad() {
    foo({});
    //~^ unit_arg
    foo3({}, 2, 2);
    //~^ unit_arg
    taking_two_units({}, foo(0));
    //~^ unit_arg
    taking_three_units({}, foo(0), foo(1));
    //~^ unit_arg
}

fn taking_two_units(a: (), b: ()) {}
fn taking_three_units(a: (), b: (), c: ()) {}

fn main() {
    bad();
}

fn issue14857() {
    let fn_take_unit = |_: ()| {};
    fn_take_unit(Default::default());
    //~^ unit_arg

    fn some_other_fn(_: &i32) {}

    macro_rules! another_mac {
        () => {
            some_other_fn(&Default::default())
        };
        ($e:expr) => {
            some_other_fn(&$e)
        };
    }

    fn_take_unit(another_mac!());
    //~^ unit_arg
    fn_take_unit(another_mac!(1));
    //~^ unit_arg

    macro_rules! mac {
        (nondef $e:expr) => {
            $e
        };
        (empty_block) => {{}};
    }
    fn_take_unit(mac!(nondef Default::default()));
    //~^ unit_arg
    fn_take_unit(mac!(empty_block));
    //~^ unit_arg

    fn def<T: Default>() -> T {
        Default::default()
    }

    fn_take_unit(def());
    //~^ unit_arg
}
