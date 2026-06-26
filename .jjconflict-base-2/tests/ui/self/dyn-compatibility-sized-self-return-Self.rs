//@ run-pass
// Check that a trait is still dyn-compatible (and usable) if it has
// methods that return `Self` so long as they require `Self : Sized`.


trait Counter {
    fn new() -> Self where Self : Sized;
    fn tick(&mut self) -> u32;
}

struct CCounter {
    c: u32
}

impl Counter for CCounter {
    fn new() -> CCounter { CCounter { c: 0 } }
    fn tick(&mut self) -> u32 { self.c += 1; self.c }
}

fn preticked<C:Counter>() -> C {
    let mut c: C = Counter::new();
    tick(&mut c);
    c
}

fn tick(c: &mut dyn Counter) {
    tick_generic(c);
}

fn tick_generic<C:?Sized+Counter>(c: &mut C) {
    c.tick();
    c.tick();
}

fn main() {
    let mut c = preticked::<CCounter>();
    tick(&mut c);
    assert_eq!(c.tick(), 5);
}
