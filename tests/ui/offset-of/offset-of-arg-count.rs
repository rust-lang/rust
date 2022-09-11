#![feature(offset_of)]

use std::mem::offset_of;

fn main() {
    offset_of!(NotEnoughArguments); //~ ERROR expected one of
    offset_of!(NotEnoughArgumentsWithAComma, ); //~ ERROR expected 2 arguments
    offset_of!(Container, field, too many arguments); //~ ERROR expected 2 arguments
}
