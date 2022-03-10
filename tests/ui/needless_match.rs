// run-rustfix
#![warn(clippy::needless_match)]
#![allow(clippy::manual_map)]
#![allow(dead_code)]

#[derive(Clone, Copy)]
enum Choice {
    A,
    B,
    C,
    D,
}

#[allow(unused_mut)]
fn useless_match() {
    let mut i = 10;
    let _: i32 = match i {
        0 => 0,
        1 => 1,
        2 => 2,
        _ => i,
    };
    let _: i32 = match i {
        0 => 0,
        1 => 1,
        ref i => *i,
    };
    let mut _i_mut = match i {
        0 => 0,
        1 => 1,
        ref mut i => *i,
    };

    let s = "test";
    let _: &str = match s {
        "a" => "a",
        "b" => "b",
        s => s,
    };
}

fn custom_type_match(se: Choice) {
    let _: Choice = match se {
        Choice::A => Choice::A,
        Choice::B => Choice::B,
        Choice::C => Choice::C,
        Choice::D => Choice::D,
    };
    // Don't trigger
    let _: Choice = match se {
        Choice::A => Choice::A,
        Choice::B => Choice::B,
        _ => Choice::C,
    };
    // Mingled, don't trigger
    let _: Choice = match se {
        Choice::A => Choice::B,
        Choice::B => Choice::C,
        Choice::C => Choice::D,
        Choice::D => Choice::A,
    };
}

fn option_match(x: Option<i32>) {
    let _: Option<i32> = match x {
        Some(a) => Some(a),
        None => None,
    };
    // Don't trigger, this is the case for manual_map_option
    let _: Option<i32> = match x {
        Some(a) => Some(-a),
        None => None,
    };
}

fn func_ret_err<T>(err: T) -> Result<i32, T> {
    Err(err)
}

fn result_match() {
    let _: Result<i32, i32> = match Ok(1) {
        Ok(a) => Ok(a),
        Err(err) => Err(err),
    };
    let _: Result<i32, i32> = match func_ret_err(0_i32) {
        Err(err) => Err(err),
        Ok(a) => Ok(a),
    };
}

fn if_let_option() -> Option<i32> {
    if let Some(a) = Some(1) { Some(a) } else { None }
}

fn if_let_result(x: Result<(), i32>) {
    let _: Result<(), i32> = if let Err(e) = x { Err(e) } else { x };
    let _: Result<(), i32> = if let Ok(val) = x { Ok(val) } else { x };
    // Input type mismatch, don't trigger
    let _: Result<(), i32> = if let Err(e) = Ok(1) { Err(e) } else { x };
}

fn if_let_custom_enum(x: Choice) {
    let _: Choice = if let Choice::A = x {
        Choice::A
    } else if let Choice::B = x {
        Choice::B
    } else if let Choice::C = x {
        Choice::C
    } else {
        x
    };
    // Don't trigger
    let _: Choice = if let Choice::A = x {
        Choice::A
    } else if true {
        Choice::B
    } else {
        x
    };
}

fn main() {}
