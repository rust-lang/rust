// run-pass
// edition:2018

use std::mem;

async fn foo() {
    let x = [0u8; 100];
    async {}.await;
    println!("{}", x.len());
}

async fn a() {
    let fut = foo();
    let fut = fut;
    fut.await;
}

async fn b() {
    let fut = foo();
    println!("{}", mem::size_of_val(&fut));
    let fut = fut;
    fut.await;
}

fn main() {
    assert_eq!(mem::size_of_val(&foo()), 102);

    // 1 + sizeof(foo)
    assert_eq!(mem::size_of_val(&a()), 103);

    // 1 + (sizeof(foo) * 2)
    assert_eq!(mem::size_of_val(&b()), 103);
}
