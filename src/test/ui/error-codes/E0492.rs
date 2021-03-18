use std::sync::atomic::AtomicUsize;

const A: AtomicUsize = AtomicUsize::new(0);
const B: &'static AtomicUsize = &A; //~ ERROR E0492
static C: &'static AtomicUsize = &A; //~ ERROR E0492

const NONE: &'static Option<AtomicUsize> = &None;

fn main() {
}
