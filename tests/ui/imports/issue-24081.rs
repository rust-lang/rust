use std::ops::Add;
use std::ops::Sub;
use std::ops::Mul;
use std::ops::Div;
use std::ops::Rem;

type Add = bool; //~ ERROR the name `Add` is defined multiple times
//~| NOTE_NONVIRAL `Add` redefined here
struct Sub { x: f32 } //~ ERROR the name `Sub` is defined multiple times
//~| NOTE_NONVIRAL `Sub` redefined here
enum Mul { A, B } //~ ERROR the name `Mul` is defined multiple times
//~| NOTE_NONVIRAL `Mul` redefined here
mod Div { } //~ ERROR the name `Div` is defined multiple times
//~| NOTE_NONVIRAL `Div` redefined here
trait Rem {  } //~ ERROR the name `Rem` is defined multiple times
//~| NOTE_NONVIRAL `Rem` redefined here

fn main() {}
