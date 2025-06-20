
#[repr(C, u8)]
pub enum Foo {
    A(bool) = b'A',
    B(char) = b'C',
}
