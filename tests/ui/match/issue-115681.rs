//@ run-pass
//@ compile-flags: -C opt-level=1

// Make sure LLVM does not miscompile this match.
fn main() {
    enum Bits {
        None = 0x00,
        Low = 0x40,
        High = 0x80,
        Both = 0xC0,
    }

    let value = Box::new(0x40u8);
    let mut out = Box::new(0u8);

    let bits = match *value {
        0x00 => Bits::None,
        0x40 => Bits::Low,
        0x80 => Bits::High,
        0xC0 => Bits::Both,
        _ => return,
    };

    match bits {
        Bits::None | Bits::Low => {
            *out = 1;
        }
        _ => (),
    }

    assert_eq!(*out, 1);
}
