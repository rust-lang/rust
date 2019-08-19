// Check that dynamically sized rvalues are forbidden

#![feature(box_syntax)]

pub fn main() {
    let _x: Box<str> = box *"hello world";
    //~^ ERROR E0161
    //~^^ ERROR cannot move out of a shared reference

    let array: &[isize] = &[1, 2, 3];
    let _x: Box<[isize]> = box *array;
    //~^ ERROR E0161
    //~^^ ERROR cannot move out of type `[isize]`, a non-copy slice
}
