//@ known-bug: #131103
struct Struct<const N: i128>(pub [u8; N]);

pub fn function(value: Struct<3>) -> u8 {
    value.0[0]
}
