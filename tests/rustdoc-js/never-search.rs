#![feature(never_type)]

#[allow(nonstandard_style)]
pub struct never;

pub fn loops() -> ! {
    loop {}
}
pub fn returns() -> never {
    never
}

pub fn impossible(x: !) {
    match x {}
}
pub fn uninteresting(x: never) {
    match x {
        never => {}
    }
}

pub fn box_impossible(x: Box<!>) {
    match *x {}
}
pub fn box_uninteresting(x: Box<never>) {
    match *x {
        never => {}
    }
}
