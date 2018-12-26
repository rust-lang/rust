// error-pattern:index out of bounds: the len is 5 but the index is 10

const C: &'static [u8; 5] = b"hello";

#[allow(const_err)]
fn test() -> u8 {
    C[10]
}

fn main() {
    test();
}
