//@ run-pass
#![allow(unused_variables)]
struct Leak<'a> {
    dropped: &'a mut bool
}

impl<'a> Drop for Leak<'a> {
    fn drop(&mut self) {
        *self.dropped = true;
    }
}

fn main() {
    let mut dropped = false;
    {
        let leak = Leak { dropped: &mut dropped };
        for ((), leaked) in Some(((), leak)).into_iter() {}
    }

    assert!(dropped);
}
