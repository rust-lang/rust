// rustfmt-style_edition: 2024

// versionsorted
pub use print0msg;
pub use printémsg;
pub use print๙msg;

fn main() {}

/// '๙' = 0E59;THAI DIGIT NINE;Nd; (Non-ASCII Decimal_Number, one string chunk)
///
/// U+0E59 > U+00E9, sorts third
mod print๙msg {}

/// '0' = 0030;DIGIT ZERO;Nd; (ASCII Decimal_Number, splits into 3 chunks ("print",0,"msg"))
///
/// shortest chunk "print", sorts first
mod print0msg {}

/// 'é' = 00E9;LATIN SMALL LETTER E WITH ACUTE;Ll; (Lowercase_Letter, one string chunk)
///
/// U+00E9 < U+0E59, sorts second
mod printémsg {}
