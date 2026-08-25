//@ run-pass


pub fn main() {
    let _quux: Box<Vec<usize>> = Box::new(Vec::new());
}
