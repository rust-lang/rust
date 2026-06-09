// Here we check that type ascription is syntactically invalid when
// not in the top position of an ascribing `let` binding or function parameter.


// This has no effect.
// We include it to demonstrate that this is the case:
#![feature(type_ascription)]

fn main() {}

fn _ok() {
    let _a @ _b: u8 = 0; // OK.
    fn _f(_a @ _b: u8) {} // OK.
}

#[cfg(false)]
fn case_1() {
    let a: u8 @ b = 0;
    //~^ ERROR expected one of `!`
}

#[cfg(false)]
fn case_2() {
    let a @ (b: u8);
    //~^ ERROR expected one of `)`
}

#[cfg(false)]
fn case_3() {
    let a: T1 @ Outer(b: T2);
    //~^ ERROR expected one of `!`
}
