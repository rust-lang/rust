//@ check-pass
//@ edition: 2024

use std::sync::Mutex;

struct Fut;
impl std::future::Future for Fut {
    type Output = ();
    fn poll(self: std::pin::Pin<&mut Self>, _cx: &mut std::task::Context<'_>) -> std::task::Poll<()> {
        std::task::Poll::Ready(())
    }
}

async fn bar(mutex: &Mutex<()>) {
    let mut guard = mutex.lock();
    loop {
        drop(guard);
        Fut.await;
        guard = mutex.lock();
    }
}

fn main() {
    fn require_send<T: Send>(_: T) {}
    require_send(bar(&Mutex::new(())));
}
