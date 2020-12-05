// run-pass

fn thing<'r>(x: &'r [isize]) -> &'r [isize] { x }

pub fn main() {
    let x = &[1,2,3];
    let y = x;
    let z = thing(x);
    assert_eq!(z[2], x[2]);
    assert_eq!(z[1], y[1]);
}
