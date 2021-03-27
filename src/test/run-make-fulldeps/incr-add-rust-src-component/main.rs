use std::sync::atomic::{AtomicUsize, Ordering};

fn main() {
    println!("Hello World");
    // This generates a panic pointing into libstd,
    // so we don't have `#[track_caller]` on `AtomicUsize::load`.
    // If `#[track_caller]` ever gets added to this method, the test
    // will start failing, and we'll need to find a new method to call
    AtomicUsize::new(0).load(Ordering::Release);
}
