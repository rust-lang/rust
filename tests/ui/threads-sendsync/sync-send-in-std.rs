//@ run-pass

//@ ignore-wasm32 networking not available
//@ ignore-sgx ToSocketAddrs cannot be used for DNS Resolution
//@ ignore-fuchsia Req. test-harness networking privileges

use std::net::ToSocketAddrs;

fn is_sync<T>(_: T)
where
    T: Sync,
{
}
fn is_send<T>(_: T)
where
    T: Send,
{
}

macro_rules! all_sync_send {
    ($ctor:expr, $($iter:ident),+) => ({
        $(
            let mut x = $ctor;
            is_sync(x.$iter());
            let mut y = $ctor;
            is_send(y.$iter());
        )+
    })
}

fn main() {
    all_sync_send!("localhost:80".to_socket_addrs().unwrap(), next);
}
