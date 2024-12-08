pub trait FromBuf<'a> {
    fn from_buf(_: &'a [u8]) -> Self;
}
