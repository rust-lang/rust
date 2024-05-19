//@ run-pass
// Checks that higher-ranked extern fn pointers implement the full range of Fn traits.

fn square(x: &isize) -> isize { (*x) * (*x) }

fn call_it<F:Fn(&isize)->isize>(f: &F, x: isize) -> isize {
    (*f)(&x)
}

fn call_it_boxed(f: &dyn Fn(&isize) -> isize, x: isize) -> isize {
    f(&x)
}

fn call_it_mut<F:FnMut(&isize)->isize>(f: &mut F, x: isize) -> isize {
    (*f)(&x)
}

fn call_it_once<F:FnOnce(&isize)->isize>(f: F, x: isize) -> isize {
    f(&x)
}

fn main() {
    let x = call_it(&square, 22);
    let x1 = call_it_boxed(&square, 22);
    let y = call_it_mut(&mut square, 22);
    let z = call_it_once(square, 22);
    assert_eq!(x, square(&22));
    assert_eq!(x1, square(&22));
    assert_eq!(y, square(&22));
    assert_eq!(z, square(&22));
}
