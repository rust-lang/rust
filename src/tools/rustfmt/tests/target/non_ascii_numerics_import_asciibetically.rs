// rustfmt-style_edition: 2015

// ascii-betically sorted
pub use print0msg;
pub use printémsg;
pub use print๙msg;

fn main() {}

/// '๙' = 0E59;THAI DIGIT NINE;Nd; (Non-ASCII Decimal_Number, sorts third)
mod print๙msg {}

/// '0' = 0030;DIGIT ZERO;Nd; (ASCII Decimal_Number, sorts first)
mod print0msg {}

/// 'é' = 00E9;LATIN SMALL LETTER E WITH ACUTE;Ll; (Lowercase_Letter, sorts second)
mod printémsg {}
