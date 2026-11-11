//@ run-pass

const X1: &'static [u8] = &[b'1'];
const X2: &'static [u8] = b"1";
const X3: &'static [u8; 1] = &[b'1'];
const X4: &'static [u8; 1] = b"1";

static Y1: u8 = X1[0];
static Y2: u8 = X2[0];
static Y3: u8 = X3[0];
static Y4: u8 = X4[0];

fn main() {
    assert_eq!(Y1, Y2);
    assert_eq!(Y1, Y3);
    assert_eq!(Y1, Y4);
}
