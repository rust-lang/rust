#![feature(box_syntax)]

fn main() {
    let y: Box<isize> = box 42;
    let mut x: Box<isize>;
    loop {
        println!("{}", y);
        loop {
            loop {
                loop {
                    x = y; //~ ERROR use of moved value
                    x.clone();
                }
            }
        }
    }
}
