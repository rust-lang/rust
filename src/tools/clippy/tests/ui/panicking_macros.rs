#![warn(clippy::unimplemented, clippy::unreachable, clippy::todo, clippy::panic)]
#![allow(clippy::assertions_on_constants)]

extern crate core;

fn panic() {
    let a = 2;
    panic!();
    panic!("message");
    panic!("{} {}", "panic with", "multiple arguments");
    let b = a + 2;
}

fn todo() {
    let a = 2;
    todo!();
    todo!("message");
    todo!("{} {}", "panic with", "multiple arguments");
    let b = a + 2;
}

fn unimplemented() {
    let a = 2;
    unimplemented!();
    unimplemented!("message");
    unimplemented!("{} {}", "panic with", "multiple arguments");
    let b = a + 2;
}

fn unreachable() {
    let a = 2;
    unreachable!();
    unreachable!("message");
    unreachable!("{} {}", "panic with", "multiple arguments");
    let b = a + 2;
}

fn core_versions() {
    use core::{panic, todo, unimplemented, unreachable};
    panic!();
    todo!();
    unimplemented!();
    unreachable!();
}

fn main() {
    panic();
    todo();
    unimplemented();
    unreachable();
    core_versions();
}
