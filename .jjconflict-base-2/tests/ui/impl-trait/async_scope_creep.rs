#![feature(type_alias_impl_trait)]
//@ edition:2021
//@ check-pass
//@ revisions: tait rpit

struct Pending {}

struct CantOpen {}

trait AsyncRead {}

impl AsyncRead for i32 {}

type PendingReader<'a> = impl AsyncRead + 'a;

#[cfg(tait)]
type OpeningReadFuture<'a> = impl std::future::Future<Output = Result<PendingReader<'a>, CantOpen>>;

impl Pending {
    async fn read(&mut self) -> Result<impl AsyncRead + '_, CantOpen> {
        Ok(42)
    }

    #[cfg(tait)]
    #[define_opaque(OpeningReadFuture)]
    fn read_fut(&mut self) -> OpeningReadFuture<'_> {
        self.read()
    }

    #[cfg(rpit)]
    #[define_opaque(PendingReader)]
    fn read_fut(
        &mut self,
    ) -> impl std::future::Future<Output = Result<PendingReader<'_>, CantOpen>> {
        self.read()
    }
}

fn main() {}
