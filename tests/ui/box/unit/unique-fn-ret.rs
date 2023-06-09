// run-pass

fn f() -> Box<isize> {
    Box::new(100)
}

pub fn main() {
    assert_eq!(f(), Box::new(100));
}
