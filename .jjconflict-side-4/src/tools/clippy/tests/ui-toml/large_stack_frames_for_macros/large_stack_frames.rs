//@ignore-target: i686
//@normalize-stderr-test: "\b10000(08|16|32)\b" -> "100$$PTR"
//@normalize-stderr-test: "\b2500(060|120)\b" -> "250$$PTR"

#![warn(clippy::large_stack_frames)]

extern crate serde;
use serde::{Deserialize, Serialize};

struct ArrayDefault<const N: usize>([u8; N]);

macro_rules! mac {
    ($name:ident) => {
        fn foo() {
            let $name = 1;
            println!("macro_name called");
        }

        fn bar() {
            let $name = ArrayDefault([0; 1000]);
        }
    };
}

mac!(something);
//~^ large_stack_frames
//~| large_stack_frames

#[derive(Deserialize, Serialize)]
//~^ large_stack_frames
//~| large_stack_frames
//~| large_stack_frames
//~| large_stack_frames
//~| large_stack_frames
//~| large_stack_frames
//~| large_stack_frames
//~| large_stack_frames
struct S {
    a: [u128; 31],
}

fn main() {}
