// Tests for "default" bounds inferred for traits with no bounds list.

trait Foo {}

fn a(_x: Box<Foo+Send>) {
}

fn b(_x: &'static (Foo+'static)) {
}

fn c(x: Box<Foo+Sync>) {
    a(x); //~ ERROR mismatched types
}

fn d(x: &'static (Foo+Sync)) {
    b(x);
}

fn main() {}
