#![warn(clippy::unimplemented, clippy::unreachable, clippy::todo, clippy::panic)]
#![allow(clippy::assertions_on_constants)]

fn panic() {
    let a = 2;
    panic!();
    let b = a + 2;
}

fn todo() {
    let a = 2;
    todo!();
    let b = a + 2;
}

fn unimplemented() {
    let a = 2;
    unimplemented!();
    let b = a + 2;
}

fn unreachable() {
    let a = 2;
    unreachable!();
    let b = a + 2;
}

fn main() {
    panic();
    todo();
    unimplemented();
    unreachable();
}
