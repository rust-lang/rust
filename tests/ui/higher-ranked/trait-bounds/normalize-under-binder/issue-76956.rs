// check-pass

use std::ops::Deref;

struct Data {
    boxed: Box<&'static i32>
}

impl Data {
    fn use_data(&self, user: impl for <'a> FnOnce(<Box<&'a i32> as Deref>::Target)) {
        user(*self.boxed)
    }
}

fn main() {}
