//@ edition:2018
//@ compile-flags: --crate-type lib

async fn no_move_across_await_tuple() -> Vec<usize> {
    let x = (vec![3], vec![4, 4]);
    drop(x.1);
    nothing().await;
    x.1
    //~^ ERROR use of moved value: `x.1`
}

async fn nothing() {}
