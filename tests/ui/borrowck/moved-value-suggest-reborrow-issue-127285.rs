//@ run-rustfix

#![allow(dead_code)]

struct X(u32);

impl X {
    fn f(&mut self) {
        generic(self);
        self.0 += 1;
        //~^ ERROR: use of moved value: `self` [E0382]
    }
}

fn generic<T>(_x: T) {}

fn main() {}
