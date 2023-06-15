// build-pass
// edition:2018

// Regression test to ensure we handle debruijn indices correctly in projection
// normalization under binders. Found in crater run for #85499

use std::future::{Future, Ready};
async fn read() {
    connect(&()).await;
}
async fn connect<A: ToSocketAddr>(addr: A) {
    addr.to_socket_addr().await;
}
pub trait ToSocketAddr {
    type Future: Future<Output = ()>;
    fn to_socket_addr(&self) -> Self::Future;
}
impl ToSocketAddr for &() {
    type Future = Ready<()>;
    fn to_socket_addr(&self) -> Self::Future {
        unimplemented!()
    }
}
struct Server;
impl Server {
    fn and_then<F>(self, _fun: F) -> AndThen<F> {
        unimplemented!()
    }
}
struct AndThen<F> {
    _marker: std::marker::PhantomData<F>,
}
pub async fn run<F>(_: F) {
}
fn main() {
    let _ = async {
        let server = Server;
        let verification_route = server.and_then(read);
        run(verification_route).await;
    };
}
