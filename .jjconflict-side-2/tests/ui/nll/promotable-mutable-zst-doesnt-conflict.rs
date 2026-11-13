// Check that mutable promoted length zero arrays don't check for conflicting
// access

//@ check-pass

pub fn main() {
    let mut x: Vec<&[i32; 0]> = Vec::new();
    for _ in 0..10 {
        x.push(&[]);
    }
}
