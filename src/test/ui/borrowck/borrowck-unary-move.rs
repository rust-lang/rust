// ignore-tidy-linelength
// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

fn foo(x: Box<isize>) -> isize {
    let y = &*x;
    free(x); //[ast]~ ERROR cannot move out of `x` because it is borrowed
    //[mir]~^ ERROR cannot move out of `x` because it is borrowed
    *y
}

fn free(_x: Box<isize>) {
}

fn main() {
}
