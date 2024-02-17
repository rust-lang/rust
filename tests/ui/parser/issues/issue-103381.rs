//@ run-rustfix

#![feature(let_chains)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(irrefutable_let_patterns)]

fn err_some(b: bool, x: Option<u32>) {
    if b && if let Some(x) = x {}
    //~^ ERROR unexpected `if` in the condition expression
}

fn err_none(b: bool, x: Option<u32>) {
    if b && if let None = x {}
    //~^ ERROR unexpected `if` in the condition expression
}

fn err_bool_1() {
    if true && if true { true } else { false };
    //~^ ERROR unexpected `if` in the condition expression
}

fn err_bool_2() {
    if true && if false { true } else { false };
    //~^ ERROR unexpected `if` in the condition expression
}

fn should_ok_1() {
    if true && if let x = 1 { true } else { true } {}
}

fn should_ok_2() {
    if true && if let 1 = 1 { true } else { true } {}
}

fn should_ok_3() {
    if true && if true { true } else { false } {}
}

fn shoule_match_ok() {
    #[cfg(feature = "full")]
    {
        let a = 1;
        let b = 2;
        if match a {
            1 if b == 1 => true,
            _ => false,
        } && if a > 1 { true } else { false }
        {
            true
        }
    }
}

fn should_ok_in_nested() {
    if true && if true { true } else { false } { true } else { false };
}

fn main() {}
