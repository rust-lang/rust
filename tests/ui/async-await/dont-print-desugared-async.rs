// Test that we don't show variables with from async fn desugaring

//@ edition:2018

async fn async_fn(&ref mut s: &[i32]) {}
//~^ ERROR cannot borrow data in a `&` reference as mutable [E0596]

fn main() {}
