//@ run-pass


fn test_simple() {
    let r = match true { true => { true } false => { panic!() } };
    assert_eq!(r, true);
}

fn test_box() {
    let r = match true { true => { vec![10] } false => { panic!() } };
    assert_eq!(r[0], 10);
}

pub fn main() { test_simple(); test_box(); }
