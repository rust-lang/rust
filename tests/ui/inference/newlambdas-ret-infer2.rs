//@ run-pass

#![allow(dead_code)]
// Test that the lambda kind is inferred correctly as a return
// expression


fn unique() -> Box<dyn FnMut()+'static> { Box::new(|| ()) }

pub fn main() {
}
