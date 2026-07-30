use std::ops::{Fn, FnMut, FnOnce};

extern "C" fn square(x: &isize) -> isize {
    (*x) * (*x)
}

fn call_it<F: ?Sized + Fn(&isize) -> isize>(f: &F, i: isize) -> isize {
    f(&i)
}

fn call_it_mut<F: ?Sized + FnMut(&isize) -> isize>(f: &mut F, i: isize) -> isize {
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

    let mut square_dyn: Box<dyn Fn(&isize) -> isize> = Box::new(square);
    assert_eq!(call_it(&*square_dyn, 28), 784);
    assert_eq!(call_it_mut(&mut *square_dyn, 29), 841);
    assert_eq!(call_it_once(square_dyn, 30), 900);
}
