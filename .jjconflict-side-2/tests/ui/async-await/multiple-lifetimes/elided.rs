//@ edition:2018
//@ run-pass

// Test that we can use async fns with multiple arbitrary lifetimes.

async fn multiple_elided_lifetimes(_: &u8, _: &u8) {}

fn main() {
    let _ = multiple_elided_lifetimes(&22, &44);
}
