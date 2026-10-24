//@ run-pass

#![allow(dead_code)]

struct X(u32);

impl X {
    fn f(&mut self) {
        generic(self);
        self.0 += 1;
    }
}

fn generic<T>(_x: T) {}

fn main() {}
