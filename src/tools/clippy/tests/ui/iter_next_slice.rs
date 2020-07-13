// run-rustfix
#![warn(clippy::iter_next_slice)]

fn main() {
    // test code goes here
    let s = [1, 2, 3];
    let v = vec![1, 2, 3];

    s.iter().next();
    // Should be replaced by s.get(0)

    s[2..].iter().next();
    // Should be replaced by s.get(2)

    v[5..].iter().next();
    // Should be replaced by v.get(5)

    v.iter().next();
    // Should be replaced by v.get(0)

    let o = Some(5);
    o.iter().next();
    // Shouldn't be linted since this is not a Slice or an Array
}
