// Check that coercions are propagated through match and if expressions.

// pretty-expanded FIXME #23616

use std::boxed::Box;

pub fn main() {
    let _: Box<[isize]> = if true { Box::new([1, 2, 3]) } else { Box::new([1]) };

    let _: Box<[isize]> = match true { true => Box::new([1, 2, 3]), false => Box::new([1]) };

    // Check we don't get over-keen at propagating coercions in the case of casts.
    let x = if true { 42 } else { 42u8 } as u16;
    let x = match true { true => 42, false => 42u8 } as u16;
}
