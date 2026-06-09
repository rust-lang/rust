//@ check-pass
//@ edition: 2021

trait TcpStack {
    type Connection<'a>: Sized where Self: 'a;
    fn connect<'a>(&'a self) -> Self::Connection<'a>;

    #[allow(async_fn_in_trait)]
    async fn async_connect<'a>(&'a self) -> Self::Connection<'a>;
}

fn main() {}
