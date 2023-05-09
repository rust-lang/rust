//@compile-flags: -Zmiri-tree-borrows
//@error-in-other-file: /deallocation through .* is forbidden/

fn inner(x: &mut i32, f: fn(&mut i32)) {
    // `f` may mutate, but it may not deallocate!
    f(x)
}

fn main() {
    inner(Box::leak(Box::new(0)), |x| {
        let raw = x as *mut _;
        drop(unsafe { Box::from_raw(raw) });
    });
}
