//@ edition:2015
#![feature(min_generic_const_args)]
#![allow(incomplete_features)]

pub enum Mode {
    Cool,
}

pub trait Parse {
    const MODE: Mode;
}

pub trait CoolStuff: Parse<MODE = Mode::Cool> {}
//~^ ERROR expected constant, found type
//~| ERROR expected constant, found type
//~| ERROR expected constant, found type
//~| ERROR expected type

fn no_help() -> Mode::Cool {}
//~^ ERROR expected type, found variant

fn main() {}
