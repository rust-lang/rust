// Regression test for #108072: do not ICE upon unmet trait alias constraint

#![feature(trait_alias)]

trait IteratorAlias = Iterator;

fn f(_: impl IteratorAlias) {}

fn main() {
    f(()) //~ ERROR the trait bound `(): IteratorAlias` is not satisfied
}
