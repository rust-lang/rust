//@ edition: 2021

#![feature(anonymous_lifetime_in_impl_trait)]

fn anon_in_apit_bound(x: &(), y: &(), mut z: impl Iterator<Item = &u8>) -> impl Sized {
    (x, y, z.next())
    //~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds
}

fn main() {}
