//! The drop checker only complains about a `#[manually_drop]` type if it _itself_ defines `Drop`.

// FIXME: this does test dropck, does it also test needs_drop?

#![feature(manually_drop_attr)]


// For example, this is absolutely fine:

#[manually_drop]
struct ManuallyDrop<T>(T);

fn drop_out_of_order_ok<T>(x: T) {
    let mut manually_dropped = ManuallyDrop(None);
    // x will be dropped before manually_dropped.
    let x = x;
    // ... but this is still fine, because it doesn't have logic on Drop.
    manually_dropped.0 = Some(&x);
}

// ... but this is not:

#[manually_drop]
struct ManuallyDropWithDestructor<T>(T);
impl<T> Drop for ManuallyDropWithDestructor<T> {
    fn drop(&mut self) {
        // maybe we read self.0 here!
    }
}

fn drop_out_of_order_not_ok<T>(x: T) {
    let mut manually_dropped_bad = ManuallyDropWithDestructor(None);
    let x = x;
    manually_dropped_bad.0 = Some(&x);
    //~^ ERROR `x` does not live long enough
}

fn main() {}
