// run-pass

// pretty-expanded FIXME #23616

pub fn main() {
    let _quux: Box<Vec<usize>> = Box::new(Vec::new());
}
