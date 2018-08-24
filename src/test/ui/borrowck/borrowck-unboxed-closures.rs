fn a<F:Fn(isize, isize) -> isize>(mut f: F) {
    let g = &mut f;
    f(1, 2);    //~ ERROR cannot borrow `f` as immutable
}

fn b<F:FnMut(isize, isize) -> isize>(f: F) {
    f(1, 2);    //~ ERROR cannot borrow immutable argument
}

fn c<F:FnOnce(isize, isize) -> isize>(f: F) {
    f(1, 2);
    f(1, 2);    //~ ERROR use of moved value
}

fn main() {}
