//@ revisions:rpass1 rpass2

#[cfg(rpass1)]
pub static A: u8 = 42;
#[cfg(rpass2)]
pub static A: u8 = 43;

static B: &u8 = &C.1;
static C: (&&u8, u8) = (&B, A);

fn main() {
    assert_eq!(*B, A);
    assert_eq!(**C.0, A);
    assert_eq!(C.1, A);
}
