// compile-fail
// edition:2018
// compile-flags: --crate-type lib

async fn no_non_guaranteed_initialization(x: usize) -> usize {
    let y;
    if x > 5 {
        y = echo(10).await;
    }
    y
    //~^ use of possibly-uninitialized variable: `y`
}

async fn echo(x: usize) -> usize { x + 1 }
