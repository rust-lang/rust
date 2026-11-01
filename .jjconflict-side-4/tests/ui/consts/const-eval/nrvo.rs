//@ run-pass

// When the NRVO is applied, the return place (`_0`) gets treated like a normal local. For example,
// its address may be taken and it may be written to indirectly. Ensure that the const-eval
// interpreter can handle this.

#[inline(never)] // Try to ensure that MIR optimizations don't optimize this away.
const fn init(buf: &mut [u8; 1024]) {
    buf[33] = 3;
    buf[444] = 4;
}

const fn nrvo() -> [u8; 1024] {
    let mut buf = [0; 1024];
    init(&mut buf);
    buf
}

const BUF: [u8; 1024] = nrvo();

fn main() {
    assert_eq!(BUF[33], 3);
    assert_eq!(BUF[19], 0);
    assert_eq!(BUF[444], 4);
}
