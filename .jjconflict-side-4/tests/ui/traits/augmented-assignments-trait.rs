//@ check-pass
use std::ops::AddAssign;

struct Int(#[allow(dead_code)] i32);

impl AddAssign for Int {
    fn add_assign(&mut self, _: Int) {
        unimplemented!()
    }
}

fn main() {}
