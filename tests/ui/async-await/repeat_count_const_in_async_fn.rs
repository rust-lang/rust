//@ check-pass
//@ edition:2018
//@ compile-flags: --crate-type=lib

pub async fn test() {
    const C: usize = 4;
    foo(&mut [0u8; C]).await;
}

async fn foo(_: &mut [u8]) {}
