#![feature(associated_const_equality)]

pub enum Mode {
    Cool,
}

pub trait Parse {
    const MODE: Mode;
}

pub trait CoolStuff: Parse<MODE = Mode::Cool> {}
//~^ ERROR expected associated constant bound
//~| ERROR expected associated constant bound
//~| ERROR expected type

fn no_help() -> Mode::Cool {}
//~^ ERROR expected type, found variant

fn main() {}
