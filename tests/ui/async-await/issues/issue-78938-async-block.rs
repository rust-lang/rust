//@ edition:2018

use std::{sync::Arc, future::Future, pin::Pin, task::{Context, Poll}};

async fn f() {
    let room_ref = Arc::new(Vec::new());

    let gameloop_handle = spawn(async { //~ ERROR E0373
        game_loop(Arc::clone(&room_ref))
    });
    gameloop_handle.await;
}

fn game_loop(v: Arc<Vec<usize>>) {}

fn spawn<F>(future: F) -> JoinHandle
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    loop {}
}

struct JoinHandle;

impl Future for JoinHandle {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        loop {}
    }
}

fn main() {}
