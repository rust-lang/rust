//@ run-pass
// Test that the call operator autoderefs when calling to an object type.

fn make_adder(x: isize) -> Box<dyn FnMut(isize)->isize + 'static> {
    Box::new(move |y| { x + y })
}

pub fn main() {
    let mut adder = make_adder(3);
    let z = adder(2);
    println!("{}", z);
    assert_eq!(z, 5);
}
