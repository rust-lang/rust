//@ build-pass
//@ edition:2018
//@ compile-flags: --crate-type lib

struct Small {
    x: Vec<usize>,
    y: Vec<usize>,
}

// You are allowed to move out part of a struct to an async fn, you still
// have access to remaining parts after awaiting
async fn move_part_await_return_rest_struct() -> Vec<usize> {
    let s = Small { x: vec![31], y: vec![19, 1441] };
    needs_vec(s.x).await;
    s.y
}

async fn needs_vec(_vec: Vec<usize>) {}
