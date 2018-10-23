// FIXME(#55223) this is just a reproduction test showing the wrong behavior

struct Slice(&'static [&'static [u8]]);

static MAP: Slice = Slice(&[
    b"CloseEvent" as &'static [u8],
]);


fn main() {}
