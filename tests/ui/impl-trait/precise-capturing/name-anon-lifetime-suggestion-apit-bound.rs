//@ edition: 2021

// regression test: a lifetime inside an APIT bound would panic the suggestion emitter
// once naming was applied to APITs, because the replacement part overlapped the APIT span.
// the expected output is the '_ fallback, which is not a correct suggestion but avoids
// the panic. the sibling test covers the cases where naming succeeds.

#![feature(anonymous_lifetime_in_impl_trait)]

fn anon_in_apit_bound(x: &(), y: &(), mut z: impl Iterator<Item = &u8>) -> impl Sized {
    (x, y, z.next())
    //~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds
}

fn main() {}
