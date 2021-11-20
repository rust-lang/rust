// check-pass

// This test checks that the `remove extra angle brackets` error doesn't happen for some
// potential edge-cases..

struct X {
    len: u32,
}

fn main() {
    let x = X { len: 3 };

    let _ = x.len > (3);

    let _ = x.len >> (3);
}
