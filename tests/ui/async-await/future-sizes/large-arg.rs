// compile-flags: -Z print-type-sizes --crate-type=lib
// edition: 2021
// build-pass
// ignore-pass

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
