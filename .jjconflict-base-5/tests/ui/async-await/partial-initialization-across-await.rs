// Test that we don't allow awaiting from an async fn while a local is partially
// initialized.

//@ edition:2018

struct S { x: i32, y: i32 }
struct T(i32, i32);

async fn noop() {}

async fn test_tuple() {
    let mut t: (i32, i32);
    t.0 = 42; //~ ERROR E0381
    noop().await;
    t.1 = 88;
    let _ = t;
}

async fn test_tuple_struct() {
    let mut t: T;
    t.0 = 42; //~ ERROR E0381
    noop().await;
    t.1 = 88;
    let _ = t;
}

async fn test_struct() {
    let mut t: S;
    t.x = 42; //~ ERROR E0381
    noop().await;
    t.y = 88;
    let _ = t;
}

fn main() {
    let _ = test_tuple();
    let _ = test_tuple_struct();
    let _ = test_struct();
}
