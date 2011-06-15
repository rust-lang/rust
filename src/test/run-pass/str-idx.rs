

fn main() {
    auto s = "hello";
    let u8 c = s.(4);
    log c;
    assert (c == 0x6f as u8);
}