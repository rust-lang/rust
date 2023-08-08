//@run-rustfix
#![warn(clippy::blocks_in_if_conditions)]
#![allow(unused, clippy::let_and_return, clippy::needless_if)]
#![warn(clippy::nonminimal_bool)]

macro_rules! blocky {
    () => {{ true }};
}

macro_rules! blocky_too {
    () => {{
        let r = true;
        r
    }};
}

fn macro_if() {
    if blocky!() {}

    if blocky_too!() {}
}

fn condition_has_block() -> i32 {
    if {
        let x = 3;
        x == 3
    } {
        6
    } else {
        10
    }
}

fn condition_has_block_with_single_expression() -> i32 {
    if { true } { 6 } else { 10 }
}

fn condition_is_normal() -> i32 {
    let x = 3;
    if true && x == 3 { 6 } else { 10 }
}

fn condition_is_unsafe_block() {
    let a: i32 = 1;

    // this should not warn because the condition is an unsafe block
    if unsafe { 1u32 == std::mem::transmute(a) } {
        println!("1u32 == a");
    }
}

fn block_in_assert() {
    let opt = Some(42);
    assert!(
        opt.as_ref()
            .map(|val| {
                let mut v = val * 2;
                v -= 1;
                v * 3
            })
            .is_some()
    );
}

fn main() {}
