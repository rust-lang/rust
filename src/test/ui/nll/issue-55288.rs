// run-pass

struct Slice(&'static [&'static [u8]]);

static MAP: Slice = Slice(&[
    b"CloseEvent" as &'static [u8],
]);

fn main() {}
