use std::sync::atomic::AtomicUsize;

const A: AtomicUsize = AtomicUsize::new(0);
static B: &'static AtomicUsize = &A; //~ ERROR E0492

fn main() {
}
