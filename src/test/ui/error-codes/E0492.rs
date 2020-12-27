#![feature(const_refs_to_cell)]

use std::sync::atomic::AtomicUsize;

const A: AtomicUsize = AtomicUsize::new(0);
const B: &'static AtomicUsize = &A; //~ ERROR E0492
// We allow this, because we would allow it *anyway* if `A` were a `static`.
static C: &'static AtomicUsize = &A;

const NONE: &'static Option<AtomicUsize> = &None;

fn main() {
}
