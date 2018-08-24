#![feature(box_syntax)]

// compile-flags: -Z emit-end-regions -Z borrowck=compare

fn dup(x: Box<isize>) -> Box<(Box<isize>,Box<isize>)> {
    box (x, x)
    //~^ use of moved value: `x` (Ast) [E0382]
    //~| use of moved value: `x` (Mir) [E0382]
}

fn main() {
    dup(box 3);
}
