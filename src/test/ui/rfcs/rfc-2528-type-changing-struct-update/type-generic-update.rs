#![feature(type_changing_struct_update)]
#![allow(incomplete_features)]

struct Machine<'a, S, M> {
    state: S,
    message: M,
    lt_str: &'a str,
    common_field: i32,
}

struct State1;
struct State2;

struct Message1;
struct Message2;

fn update() {
    let m1: Machine<State1, Message1> = Machine {
        state: State1,
        message: Message1,
        lt_str: "hello",
        common_field: 2,
    };
    // single type update
    let m2: Machine<State2, Message1> = Machine {
        state: State2,
        ..m1
    };
    // multiple type update
    let m3: Machine<State2, Message2> = Machine {
        state: State2,
        message: Message2,
        ..m1
    };
}

fn fail_update() {
    let m1: Machine<f64, f64> = Machine {
        state: 3.2,
        message: 6.4,
        lt_str: "hello",
        common_field: 2,
    };
    // single type update fail
    let m2: Machine<i32, f64> = Machine {
        ..m1
        //~^ ERROR mismatched types [E0308]
    };
    // multiple type update fail
    let m3 = Machine::<i32, i32> {
        ..m1
        //~^ ERROR mismatched types [E0308]
        //~| ERROR mismatched types [E0308]
    };
}

fn main() {}
