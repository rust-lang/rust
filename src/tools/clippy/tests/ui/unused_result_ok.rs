//@aux-build:proc_macros.rs
#![warn(clippy::unused_result_ok)]
#![allow(dead_code)]

#[macro_use]
extern crate proc_macros;

fn bad_style(x: &str) {
    x.parse::<u32>().ok();
}

fn good_style(x: &str) -> Option<u32> {
    x.parse::<u32>().ok()
}

#[rustfmt::skip]
fn strange_parse(x: &str) {
    x   .   parse::<i32>()   .   ok   ();
}

macro_rules! v {
    () => {
        Ok::<(), ()>(())
    };
}

macro_rules! w {
    () => {
        Ok::<(), ()>(()).ok();
    };
}

fn main() {
    v!().ok();
    w!();

    external! {
        Ok::<(),()>(()).ok();
    };
}
