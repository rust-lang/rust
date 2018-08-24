#![feature(box_syntax)]

fn main() {
    let _x: Box<str> = box *"hello"; //~ ERROR E0161
                                     //~^ ERROR E0507
}
