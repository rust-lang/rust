//@ build-pass
//@ edition:2018
//@ compile-flags: --crate-type lib

async fn move_part_await_return_rest_tuple() -> Vec<usize> {
    let x = (vec![3], vec![4, 4]);
    drop(x.1);
    echo(x.0[0]).await;
    x.0
}

async fn echo(x: usize) -> usize { x }
