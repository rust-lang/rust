// run-pass

pub fn main() {
    let bar: Box<_> = Box::new(3);
    let h = || -> isize { *bar };
    assert_eq!(h(), 3);
}
