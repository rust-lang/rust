#![allow(unused, clippy::diverging_sub_expression)]
#![warn(clippy::nonminimal_bool)]

fn main() {
    let a: bool = unimplemented!();
    let b: bool = unimplemented!();
    let c: bool = unimplemented!();
    let d: bool = unimplemented!();
    let e: bool = unimplemented!();
    let _ = !true;
    let _ = !false;
    let _ = !!a;
    let _ = false || a;
    // don't lint on cfgs
    let _ = cfg!(you_shall_not_not_pass) && a;
    let _ = a || !b || !c || !d || !e;
    let _ = !(!a && b);
    let _ = !(!a || b);
    let _ = !a && !(b && c);
}

fn equality_stuff() {
    let a: i32 = unimplemented!();
    let b: i32 = unimplemented!();
    let c: i32 = unimplemented!();
    let d: i32 = unimplemented!();
    let _ = a == b && c == 5 && a == b;
    let _ = a == b || c == 5 || a == b;
    let _ = a == b && c == 5 && b == a;
    let _ = a != b || !(a != b || c == d);
    let _ = a != b && !(a != b && c == d);
}

fn issue3847(a: u32, b: u32) -> bool {
    const THRESHOLD: u32 = 1_000;

    if a < THRESHOLD && b >= THRESHOLD || a >= THRESHOLD && b < THRESHOLD {
        return false;
    }
    true
}

fn issue4548() {
    fn f(_i: u32, _j: u32) -> u32 {
        unimplemented!();
    }

    let i = 0;
    let j = 0;

    if i != j && f(i, j) != 0 || i == j && f(i, j) != 1 {}
}
