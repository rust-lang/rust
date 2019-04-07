// Check that mutable promoted length zero arrays don't check for conflicting
// access

// run-pass

pub fn main() {
    let mut x: Vec<&[i32; 0]> = Vec::new();
    for i in 0..10 {
        x.push(&[]);
    }
}
