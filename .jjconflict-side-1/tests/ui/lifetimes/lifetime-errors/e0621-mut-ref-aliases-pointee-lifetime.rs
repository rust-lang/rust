// Regression test for #156682
// E0621 should not suggest `&'a mut Buffer<'a>`, which would alias the outer
// reference's lifetime with one already used inside the pointee.

pub struct Buffer<'a> {
    buf: &'a mut [u8],
}

pub fn foo<'a>(buffer: &mut Buffer<'a>) {
    buffer.buf = &mut buffer.buf[..];
    //~^ ERROR explicit lifetime required in the type of `buffer`
}

pub struct Wrapper<'a> {
    inner: Buffer<'a>,
}

pub fn baz<'a>(wrapper: &mut Wrapper<'a>) {
    wrapper.inner.buf = &mut wrapper.inner.buf[..];
    //~^ ERROR explicit lifetime required in the type of `wrapper`
}

fn main() {}
