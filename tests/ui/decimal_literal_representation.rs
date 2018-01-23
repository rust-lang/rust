


#[warn(decimal_literal_representation)]
#[allow(unused_variables)]
fn main() {
    // Hex:      7F,  80, 100, 1FF,  800,  FFA,   F0F3,     7F0F_F00D
    let good = (127, 128, 256, 511, 2048, 4090, 61_683, 2_131_750_925);
    let bad = (        // Hex:
        4096,          // 0x1000
        16_371,        // 0x3FF3
        32_773,        // 0x8005
        65_280,        // 0xFF00
        2_131_750_927, // 0x7F0F_F00F
        2_147_483_647, // 0x7FFF_FFFF
        4_042_322_160, // 0xF0F0_F0F0
    );
}
