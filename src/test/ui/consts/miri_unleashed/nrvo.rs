// compile-flags: -Zunleash-the-miri-inside-of-you
// run-pass

const fn init(buf: &mut [u8; 1024]) {
    buf[33] = 3;
    buf[444] = 4;
}

const fn nrvo(init: fn(&mut [u8; 1024])) -> [u8; 1024] {
    let mut buf = [0; 1024];
    init(&mut buf);
    buf
}

// When the NRVO is applied, the return place (`_0`) gets treated like a normal local. For example,
// its address may be taken and it may be written to indirectly. Ensure that MIRI can handle this.
const BUF: [u8; 1024] = nrvo(init);

fn main() {
    assert_eq!(BUF[33], 3);
    assert_eq!(BUF[19], 0);
    assert_eq!(BUF[444], 4);
}
