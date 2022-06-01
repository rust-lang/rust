// error-pattern: deallocating while item is protected

use std::cell::Cell;

// Check that even `&Cell` are dereferenceable.
// Also see <https://github.com/rust-lang/rust/issues/55005>.
fn inner(x: &Cell<i32>, f: fn(&Cell<i32>)) {
    // `f` may mutate, but it may not deallocate!
    f(x)
}

fn main() {
    inner(Box::leak(Box::new(Cell::new(0))), |x| {
        let raw = x as *const _ as *mut Cell<i32>;
        drop(unsafe { Box::from_raw(raw) });
    });
}
