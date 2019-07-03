// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
struct A;
impl Drop for A {
    fn drop(&mut self) {}
}

const FOO: A = A;

fn main() {}
