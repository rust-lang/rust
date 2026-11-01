#[warn(clippy::decimal_literal_representation)]
#[allow(unused_variables)]
#[rustfmt::skip]
fn main() {
    let good = (       // Hex:
        127,           // 0x7F
        256,           // 0x100
        511,           // 0x1FF
        2048,          // 0x800
        4090,          // 0xFFA
        16_371,        // 0x3FF3
        61_683,        // 0xF0F3
        2_131_750_925, // 0x7F0F_F00D
    );
    let bad = (        // Hex:
        32_773,        // 0x8005
        //~^ decimal_literal_representation
        65_280,        // 0xFF00
        //~^ decimal_literal_representation
        2_131_750_927, // 0x7F0F_F00F
        //~^ decimal_literal_representation
        2_147_483_647, // 0x7FFF_FFFF
        //~^ decimal_literal_representation
        #[allow(overflowing_literals)]
        4_042_322_160, // 0xF0F0_F0F0
        //~^ decimal_literal_representation
        32_773usize,   // 0x8005_usize
        //~^ decimal_literal_representation
        2_131_750_927isize, // 0x7F0F_F00F_isize
        //~^ decimal_literal_representation
    );
}
