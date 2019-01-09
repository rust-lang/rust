// run-pass
#![feature(lang_items)]

fn a<F:Fn(isize, isize) -> isize>(f: F) -> isize {
    f(1, 2)
}

fn b<F:FnMut(isize, isize) -> isize>(mut f: F) -> isize {
    f(3, 4)
}

fn c<F:FnOnce(isize, isize) -> isize>(f: F) -> isize {
    f(5, 6)
}

fn main() {
    let z: isize = 7;
    assert_eq!(a(move |x: isize, y| x + y + z), 10);
    assert_eq!(b(move |x: isize, y| x + y + z), 14);
    assert_eq!(c(move |x: isize, y| x + y + z), 18);
}
