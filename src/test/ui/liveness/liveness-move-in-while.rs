#![feature(box_syntax)]

fn main() {
    let y: Box<isize> = box 42;
    let mut x: Box<isize>;
    loop {
        println!("{}", y); //~ ERROR use of moved value: `y`
        while true { while true { while true { x = y; x.clone(); } } }
        //~^ ERROR use of moved value: `y`
    }
}
