//@ check-pass

trait AsBufferView {
    type Device;
}

trait Error {
    type Span;
}

trait Foo {
    type Error: Error;
    fn foo(&self) -> &<Self::Error as Error>::Span;
}

impl<D: Error, VBuf0> Foo for VBuf0
where
    VBuf0: AsBufferView<Device = D>,
{
    type Error = D;
    fn foo(&self) -> &<Self::Error as Error>::Span {
        todo!()
    }
}

fn main() {}
