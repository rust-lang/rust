#![feature(core_intrinsics)]

use std::io::Write;

fn main() {
    assert_eq!((1u128 + 2) as u16, 3);
}

#[derive(PartialEq)]
enum LoopState {
    Continue(()),
    Break(())
}

pub enum Instruction {
    Increment,
    Loop,
}

fn map(a: Option<(u8, Box<Instruction>)>) -> Option<Box<Instruction>> {
    match a {
        None => None,
        Some((_, instr)) => Some(instr),
    }
}
