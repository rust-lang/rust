#![feature(box_syntax)]

fn main() {
    let x: Box<isize> = box 0;

    println!("{}", x + 1);
    //~^ ERROR binary operation `+` cannot be applied to type `std::boxed::Box<isize>`
}
