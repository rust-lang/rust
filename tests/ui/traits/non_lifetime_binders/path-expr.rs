// check-pass

#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

fn b() where for<const C: usize> [(); C]: Clone {}

fn main() {
    b();
}
