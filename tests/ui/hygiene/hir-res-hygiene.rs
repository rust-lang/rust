//@ check-pass
//@ edition:2018
//@ aux-build:not-libstd.rs

// Check that paths created in HIR are not affected by in scope names.

extern crate not_libstd as std;

async fn the_future() {
    async {}.await;
}

fn main() -> Result<(), ()> {
    for i in 0..10 {}
    for j in 0..=10 {}
    Ok(())?;
    Ok(())
}
