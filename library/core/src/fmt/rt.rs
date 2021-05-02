pub mod v1;

pub const CMD_ARG: usize = usize::reverse_bits(0b11);
pub const CMD_FLAGS: usize = usize::reverse_bits(0b1);
pub const CMD_FILL: usize = usize::reverse_bits(0b1) + 1;
pub const CMD_WIDTH: usize = usize::reverse_bits(0b1) + 2;
pub const CMD_WIDTH_ARG: usize = usize::reverse_bits(0b1) + 3;
pub const CMD_PRECISION: usize = usize::reverse_bits(0b1) + 4;
pub const CMD_PRECISION_ARG: usize = usize::reverse_bits(0b1) + 5;
