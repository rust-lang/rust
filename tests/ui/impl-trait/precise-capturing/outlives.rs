//@ check-pass
//@ edition: 2024

// Show that precise captures allow us to skip a lifetime param for outlives

fn hello<'a: 'a, 'b: 'b>() -> impl Sized + use<'a> { }

fn outlives<'a, T: 'a>(_: T) {}

fn test<'a, 'b>() {
    outlives::<'a, _>(hello::<'a, 'b>());
}

fn main() {}
