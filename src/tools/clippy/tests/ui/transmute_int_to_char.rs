#![warn(clippy::transmute_int_to_char)]
#![allow(clippy::missing_transmute_annotations, unnecessary_transmutes)]

fn int_to_char() {
    let _: char = unsafe { std::mem::transmute(0_u32) };
    //~^ transmute_int_to_char

    let _: char = unsafe { std::mem::transmute(0_i32) };
    //~^ transmute_int_to_char

    // These shouldn't warn
    const _: char = unsafe { std::mem::transmute(0_u32) };
    const _: char = unsafe { std::mem::transmute(0_i32) };
}

fn main() {}
