//@ check-pass

fn main() {
    let array = [0x42u8; 10];
    for b in &array {
        let lo = b & 0xf;
        let hi = (b >> 4) & 0xf;
    }
}
