#![feature(box_syntax)]

fn dup(x: Box<isize>) -> Box<(Box<isize>,Box<isize>)> {
    box (x, x)
    //~^ use of moved value: `x` [E0382]
}

fn main() {
    dup(box 3);
}
