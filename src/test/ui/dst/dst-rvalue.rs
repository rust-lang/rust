// Check that dynamically sized rvalues are forbidden

#![feature(box_syntax)]

pub fn main() {
    let _x: Box<str> = box *"hello world";
    //~^ ERROR E0277

    let array: &[isize] = &[1, 2, 3];
    let _x: Box<[isize]> = box *array;
    //~^ ERROR E0277
}
