//@ run-rustfix
fn main() {
    let _ = Option:Some(vec![0, 1]); //~ ERROR path separator must be a double colon
}
