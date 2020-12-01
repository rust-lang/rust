#![allow(unused, clippy::many_single_char_names, clippy::diverging_sub_expression)]
#![warn(clippy::logic_bug)]

fn main() {
    let a: bool = unimplemented!();
    let b: bool = unimplemented!();
    let c: bool = unimplemented!();
    let d: bool = unimplemented!();
    let e: bool = unimplemented!();
    let _ = a && b || a;
    let _ = !(a && b);
    let _ = false && a;
    // don't lint on cfgs
    let _ = cfg!(you_shall_not_not_pass) && a;
    let _ = a || !b || !c || !d || !e;
    let _ = !(a && b || c);
}

fn equality_stuff() {
    let a: i32 = unimplemented!();
    let b: i32 = unimplemented!();
    let _ = a == b && a != b;
    let _ = a < b && a >= b;
    let _ = a > b && a <= b;
    let _ = a > b && a == b;
}
