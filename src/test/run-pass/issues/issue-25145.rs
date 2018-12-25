// run-pass

struct S;

impl S {
    const N: usize = 3;
}

static STUFF: [u8; S::N] = [0; S::N];

fn main() {
    assert_eq!(STUFF, [0; 3]);
}
