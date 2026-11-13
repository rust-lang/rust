// Make sure negation happens correctly. Also included:
// issue: rust-lang/rust#124583
//@ run-pass

#![feature(f16)]
#![feature(f128)]

fn main() {
    assert_eq!(0.0_f16.to_bits(), 0x0000);
    assert_eq!((-0.0_f16).to_bits(), 0x8000);
    assert_eq!(10.0_f16.to_bits(), 0x4900);
    assert_eq!((-10.0_f16).to_bits(), 0xC900);
    assert_eq!((-(-0.0f16)).to_bits(), 0x0000);

    assert_eq!(0.0_f128.to_bits(), 0x0000_0000_0000_0000_0000_0000_0000_0000);
    assert_eq!((-0.0_f128).to_bits(), 0x8000_0000_0000_0000_0000_0000_0000_0000);
    assert_eq!(10.0_f128.to_bits(), 0x4002_4000_0000_0000_0000_0000_0000_0000);
    assert_eq!((-10.0_f128).to_bits(), 0xC002_4000_0000_0000_0000_0000_0000_0000);
    assert_eq!((-(-0.0f128)).to_bits(), 0x0000_0000_0000_0000_0000_0000_0000_0000);
}
