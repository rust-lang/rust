// FIXME(#61117): Respect debuginfo-level-tests, do not force debuginfo=0
//@ compile-flags: -C debuginfo=0
//@ compile-flags: -C panic=abort -Z print-type-sizes --crate-type=lib
//@ needs-deterministic-layouts
//@ edition: 2021
//@ build-pass
//@ ignore-pass
//@ only-x86_64

pub async fn test() {
    let _ = a([0u8; 1024]).await;
}

pub async fn a<T>(t: T) -> T {
    b(t).await
}
async fn b<T>(t: T) -> T {
    c(t).await
}
async fn c<T>(t: T) -> T {
    t
}
