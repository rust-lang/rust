//@ run-pass
#[deny(warnings)]

pub fn main() {
    let _ = b"x" as &[u8];
    let _ = b"y" as &[u8; 1];
    let _ = b"z" as *const u8;
    let _ = "ä" as *const str;
}
