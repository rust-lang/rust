// https://github.com/rust-lang/rust/issues/55223

struct Slice(&'static [&'static [u8]]);

static MAP: Slice = Slice(&[
    b"CloseEvent" as &'static [u8],
]);


fn main() {}
