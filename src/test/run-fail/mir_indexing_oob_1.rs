// error-pattern:index out of bounds: the len is 5 but the index is 10

const C: [u32; 5] = [0; 5];

#[allow(const_err)]
fn test() -> u32 {
    C[10]
}

fn main() {
    test();
}
