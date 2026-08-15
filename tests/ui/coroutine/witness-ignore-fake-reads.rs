//@ check-pass
//@ edition: 2021

// regression test for #117059
struct SendNotSync(*const ());
unsafe impl Send for SendNotSync {}
// impl !Sync for SendNotSync {} // automatically disabled

struct Inner {
    stream: SendNotSync,
    state: bool,
}

struct SendSync;
impl std::ops::Deref for SendSync {
    type Target = Inner;
    fn deref(&self) -> &Self::Target {
        todo!();
    }
}

async fn next() {
    let inner = SendSync;
    match inner.state {
        true if false => {}
        false => async {}.await,
        _ => {}
    }
}

fn is_send<T: Send>(_: T) {}
fn main() {
    is_send(next())
}
