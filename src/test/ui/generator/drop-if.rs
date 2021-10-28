// build-pass

// This test case is reduced from src/test/ui/drop/dynamic-drop-async.rs

#![feature(generators)]

struct Ptr;
impl<'a> Drop for Ptr {
    fn drop(&mut self) {
    }
}

fn main() {
    let arg = true;
    let _ = || {
        let arr = [Ptr];
        if arg {
            drop(arr);
        }
        yield
    };
}
