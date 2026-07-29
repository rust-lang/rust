use std::ops::{Fn, FnMut, FnOnce};

extern "C" fn square(x: &isize) -> isize {
    (*x) * (*x)
}

fn call_it<F: Fn(&isize) -> isize>(f: &F, i: isize) -> isize {
    f(&i)
}
fn call_it_mut<F: FnMut(&isize) -> isize>(f: &mut F, i: isize) -> isize {
    f(&i)
}
fn call_it_once<F: FnOnce(&isize) -> isize>(f: F, i: isize) -> isize {
    f(&i)
}

fn main() {
    assert_eq!(call_it(&square, 22), 484);
    assert_eq!(call_it_mut(&mut square, 23), 529);
    assert_eq!(call_it_once(square, 24), 576);

    let mut square_ptr: extern "C" fn(&isize) -> isize = square;

    assert_eq!(call_it(&square_ptr, 25), 625);
    assert_eq!(call_it_mut(&mut square_ptr, 26), 676);
    assert_eq!(call_it_once(square_ptr, 27), 729);
}
