// run-pass
const C1: i32 = 0x12345678;
const C2: isize = C1 as i16 as isize;

enum E {
    V = C2
}

fn main() {
    assert_eq!(C2 as u64, E::V as u64);
}
