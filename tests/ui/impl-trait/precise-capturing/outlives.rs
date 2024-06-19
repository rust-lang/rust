//@ check-pass

// Show that precise captures allow us to skip a lifetime param for outlives

#![feature(lifetime_capture_rules_2024, precise_capturing)]

fn hello<'a: 'a, 'b: 'b>() -> impl Sized + use<'a> { }

fn outlives<'a, T: 'a>(_: T) {}

fn test<'a, 'b>() {
    outlives::<'a, _>(hello::<'a, 'b>());
}

fn main() {}
