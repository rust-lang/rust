#![warn(clippy::precedence_bits)]
#![expect(clippy::eq_op, clippy::identity_op, clippy::no_effect, clippy::precedence)]

macro_rules! trip {
    ($a:expr) => {
        match $a & 0b1111_1111u8 {
            0 => println!("a is zero ({})", $a),
            _ => println!("a is {}", $a),
        }
    };
}

fn main() {
    1 << 2 + 3;
    1 + 2 << 3;
    4 >> 1 + 1;
    1 + 3 >> 2;
    1 ^ 1 - 1;
    3 | 2 - 1;
    3 & 5 - 2;
    0x0F00 & 0x00F0 << 4;
    //~^ precedence_bits
    0x0F00 & 0xF000 >> 4;
    //~^ precedence_bits
    0x0F00 << 1 ^ 3;
    //~^ precedence_bits
    0x0F00 << 1 | 2;
    //~^ precedence_bits

    let b = 3;
    trip!(b * 8);
}
