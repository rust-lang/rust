// run-pass
// compile-flags: --edition=2018 -Zdrop-tracking-mir=yes

use std::future::{Future, Ready};
async fn read() {}
async fn connect<A: ToSocketAddr>(addr: A) {
    let _ = addr.to_socket_addr().await;
}
pub trait ToSocketAddr {
    type Future: Future;
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
    let _ = connect(&()).await;
}
fn main() {
    let _ = async {
        let server = Server;
        let verification_route = server.and_then(read);
        run(verification_route).await;
    };
}
