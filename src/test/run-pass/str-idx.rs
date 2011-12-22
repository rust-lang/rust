

fn main() {
    let s = "hello";
    let c: u8 = s[4];
    log_full(core::debug, c);
    assert (c == 0x6f as u8);
}
