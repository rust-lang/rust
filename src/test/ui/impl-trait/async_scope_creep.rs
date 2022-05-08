#![feature(type_alias_impl_trait)]
// edition:2021
// check-pass

struct Pending {}

struct CantOpen {}

trait AsyncRead {}

impl AsyncRead for i32 {}

type PendingReader<'a> = impl AsyncRead + 'a;

type OpeningReadFuture<'a> =
    impl std::future::Future<Output = Result<PendingReader<'a>, CantOpen>>;

impl Pending {
    async fn read(&mut self) -> Result<impl AsyncRead + '_, CantOpen> {
        Ok(42)
    }

    fn read_fut(&mut self) -> OpeningReadFuture<'_> {
        self.read()
    }
}

fn main() {}
