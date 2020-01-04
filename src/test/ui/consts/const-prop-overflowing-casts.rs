// build-fail
// ignore-tidy-linelength

fn main() {
    let _ = 0u8 as u32;
    let _ = (1u32 << 31) as u16; //~ ERROR truncating cast: the value 2147483648 requires 32 bits but the target type is only 16 bits
    let _ = (1u16 << 15) as u8; //~ ERROR truncating cast: the value 32768 requires 16 bits but the target type is only 8 bits
    let _ = (!0u16) as u8; //~ ERROR truncating cast: the value 65535 requires 16 bits but the target type is only 8 bits
}
