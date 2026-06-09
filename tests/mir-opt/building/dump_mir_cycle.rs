//@ compile-flags: -Zmir-opt-level=0

#[derive(Debug)]
pub struct Thing {
    pub next: &'static Thing,
}

pub static THING: Thing = Thing { next: &THING };
// CHECK: alloc{{.+}} (static: THING)

const fn thing() -> &'static Thing {
    &MUTUALLY_RECURSIVE
}

pub static MUTUALLY_RECURSIVE: Thing = Thing { next: thing() };
// CHECK: alloc{{.+}} (static: MUTUALLY_RECURSIVE)

fn main() {
    // Generate optimized MIR for the const fn, too.
    thing();
}
