// check-pass

#![feature(type_alias_impl_trait)]

struct Output;

trait Service {
    type OutputStream;

    fn stream<'l, 'a>(&'l self) -> Self::OutputStream
    where
        Self: 'a,
        'l: 'a;
}

trait Stream {
    type Item;
}

struct ImplStream<F: Fn()>(F);

impl<F: Fn()> Stream for ImplStream<F> {
    type Item = Output;
}

impl Service for () {
    type OutputStream = impl Stream<Item = Output>;

    fn stream<'l, 'a>(&'l self) -> Self::OutputStream
    where
        Self: 'a,
        'l: 'a,
    {
        ImplStream(|| ())
    }
}

fn main() {}
