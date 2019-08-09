// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// Test to make sure that explicit self params work inside closures

// pretty-expanded FIXME #23616

struct Box {
    x: usize
}

impl Box {
    pub fn set_many(&mut self, xs: &[usize]) {
        for x in xs { self.x = *x; }
    }
}

pub fn main() {}
