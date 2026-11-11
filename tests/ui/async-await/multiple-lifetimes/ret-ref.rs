//@ edition:2018

// Test that we get the expected borrow check errors when an async
// function (which takes multiple lifetimes) only returns data from
// one of them.

async fn multiple_named_lifetimes<'a, 'b>(a: &'a u8, _: &'b u8) -> &'a u8 {
    a
}

// Both are borrowed whilst the future is live.
async fn future_live() {
    let mut a = 22;
    let mut b = 44;
    let future = multiple_named_lifetimes(&a, &b);
    a += 1; //~ ERROR cannot assign
    b += 1; //~ ERROR cannot assign
    let p = future.await;
    drop(p);
}

// Just the return value is live after future is awaited.
async fn just_return_live() {
    let mut a = 22;
    let mut b = 44;
    let future = multiple_named_lifetimes(&a, &b);
    let p = future.await;
    a += 1; //~ ERROR cannot assign
    b += 1;
    drop(p);
}

// Once `p` is dead, both `a` and `b` are unborrowed.
async fn after_both_dead() {
    let mut a = 22;
    let mut b = 44;
    let future = multiple_named_lifetimes(&a, &b);
    let p = future.await;
    drop(p);
    a += 1;
    b += 1;
}

fn main() { }
