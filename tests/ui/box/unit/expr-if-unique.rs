// run-pass

// Tests for if as expressions returning boxed types
fn test_box() {
    let rs: Box<_> = if true { Box::new(100) } else { Box::new(101) };
    assert_eq!(*rs, 100);
}

pub fn main() { test_box(); }
