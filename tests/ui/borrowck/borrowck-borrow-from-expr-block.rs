// run-pass

fn borrow<F>(x: &isize, f: F) where F: FnOnce(&isize) {
    f(x)
}

fn test1(x: &Box<isize>) {
    borrow(&*(*x).clone(), |p| {
        let x_a = &**x as *const isize;
        assert!((x_a as usize) != (p as *const isize as usize));
        assert_eq!(unsafe{*x_a}, *p);
    })
}

pub fn main() {
    test1(&Box::new(22));
}
