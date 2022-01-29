#![feature(box_syntax)]

fn main() {
    let x: Box<isize> = box 0;

    println!("{}", x + 1);
    //~^ ERROR cannot add `{integer}` to `Box<isize>`
}
