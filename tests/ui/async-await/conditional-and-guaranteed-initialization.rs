//@ check-pass
//@ edition:2018
//@ compile-flags: --crate-type lib

async fn conditional_and_guaranteed_initialization(x: usize) -> usize {
    let y;
    if x > 5 {
        y = echo(10).await;
    } else {
        y = get_something().await;
    }
    y
}

async fn echo(x: usize) -> usize { x }
async fn get_something() -> usize { 10 }
