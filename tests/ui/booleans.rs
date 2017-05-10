#![feature(plugin)]
#![plugin(clippy)]
#![deny(nonminimal_bool, logic_bug)]

#[allow(unused, many_single_char_names)]
fn main() {
    let a: bool = unimplemented!();
    let b: bool = unimplemented!();
    let c: bool = unimplemented!();
    let d: bool = unimplemented!();
    let e: bool = unimplemented!();
    let _ = a && b || a;
    let _ = !(a && b);
    let _ = !true;
    let _ = !false;
    let _ = !!a;
    let _ = false && a;
    let _ = false || a;
    // don't lint on cfgs
    let _ = cfg!(you_shall_not_not_pass) && a;
    let _ = a || !b || !c || !d || !e;
    let _ = !(a && b || c);
    let _ = !(!a && b);
}

#[allow(unused, many_single_char_names)]
fn equality_stuff() {
    let a: i32 = unimplemented!();
    let b: i32 = unimplemented!();
    let c: i32 = unimplemented!();
    let d: i32 = unimplemented!();
    let e: i32 = unimplemented!();
    let _ = a == b && a != b;
    let _ = a == b && c == 5 && a == b;
    let _ = a == b && c == 5 && b == a;
    let _ = a < b && a >= b;
    let _ = a > b && a <= b;
    let _ = a > b && a == b;
    let _ = a != b || !(a != b || c == d);
}
