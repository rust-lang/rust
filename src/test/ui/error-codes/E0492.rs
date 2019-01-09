use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT};

const A: AtomicUsize = ATOMIC_USIZE_INIT;
static B: &'static AtomicUsize = &A; //~ ERROR E0492

fn main() {
}
