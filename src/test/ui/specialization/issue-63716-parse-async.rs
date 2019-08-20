// Ensure that `default async fn` will parse.
// See issue #63716 for details.

// check-pass
// edition:2018

#![feature(specialization)]

fn main() {}

#[cfg(FALSE)]
impl Foo for Bar {
    default async fn baz() {}
}
