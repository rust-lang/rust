//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows

//@[stack]error-in-other-file: protected
//@[tree]error-in-other-file: forbidden

// A reference in a closure is treated like a reference in a struct.
fn main() {
    fn invoke(f: impl FnOnce()) {
        // The closure has captured a reference that will be freed while `invoke` runs.
        f()
    }

    let p = Box::leak(Box::new(0i32));
    invoke(move || {
        drop(unsafe { Box::from_raw(p) });
    });
}
