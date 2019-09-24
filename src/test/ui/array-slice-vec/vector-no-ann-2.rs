// run-pass

// pretty-expanded FIXME #23616

#![feature(box_syntax)]

pub fn main() { let _quux: Box<Vec<usize>> = box Vec::new(); }
