//@ check-pass
//@ edition:2018

async fn foo(x: &[Vec<u32>]) -> u32 {
    0
}

async fn bar() {
    foo(&[vec![123]]).await;
}

fn main() { }
