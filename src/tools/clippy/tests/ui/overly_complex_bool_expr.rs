#![allow(unused, clippy::diverging_sub_expression)]
#![warn(clippy::overly_complex_bool_expr)]

fn main() {
    let a: bool = unimplemented!();
    let b: bool = unimplemented!();
    let c: bool = unimplemented!();
    let d: bool = unimplemented!();
    let e: bool = unimplemented!();
    let _ = a && b || a;
    //~^ overly_complex_bool_expr

    let _ = !(a && b);
    let _ = false && a;
    //~^ overly_complex_bool_expr

    // don't lint on cfgs
    let _ = cfg!(you_shall_not_not_pass) && a;
    let _ = a || !b || !c || !d || !e;
    let _ = !(a && b || c);
}

fn equality_stuff() {
    let a: i32 = unimplemented!();
    let b: i32 = unimplemented!();
    let _ = a == b && a != b;
    //~^ overly_complex_bool_expr

    let _ = a < b && a >= b;
    //~^ overly_complex_bool_expr

    let _ = a > b && a <= b;
    //~^ overly_complex_bool_expr

    let _ = a > b && a == b;
}

fn check_expect() {
    let a: i32 = unimplemented!();
    let b: i32 = unimplemented!();
    #[expect(clippy::overly_complex_bool_expr)]
    let _ = a < b && a >= b;
}

#[allow(clippy::never_loop)]
fn check_never_type() {
    loop {
        _ = (break) || true;
    }
    loop {
        _ = (return) || true;
    }
}
