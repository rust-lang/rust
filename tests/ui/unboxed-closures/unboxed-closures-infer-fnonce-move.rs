// run-pass
// Test that we are able to infer a suitable kind for this `move`
// closure that is just called (`FnOnce`).

use std::mem;

struct DropMe<'a>(&'a mut i32);

impl<'a> Drop for DropMe<'a> {
    fn drop(&mut self) {
        *self.0 += 1;
    }
}

fn main() {
    let mut counter = 0;

    {
        let drop_me = DropMe(&mut counter);
        let tick = move || mem::drop(drop_me);
        tick();
    }

    assert_eq!(counter, 1);
}
