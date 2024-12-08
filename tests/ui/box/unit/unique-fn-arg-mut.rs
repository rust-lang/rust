//@ run-pass

fn f(i: &mut Box<isize>) {
    *i = Box::new(200);
}

pub fn main() {
    let mut i = Box::new(100);
    f(&mut i);
    assert_eq!(*i, 200);
}
