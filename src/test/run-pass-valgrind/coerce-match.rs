// Check that coercions are propagated through match and if expressions.

// pretty-expanded FIXME #23616

#![feature(box_syntax)]

pub fn main() {
    let _: Box<[isize]> =
        if true { let b: Box<_> = box [1, 2, 3]; b } else { let b: Box<_> = box [1]; b };

    let _: Box<[isize]> = match true {
        true => { let b: Box<_> = box [1, 2, 3]; b }
        false => { let b: Box<_> = box [1]; b }
    };

    // Check we don't get over-keen at propagating coercions in the case of casts.
    let x = if true { 42 } else { 42u8 } as u16;
    let x = match true { true => 42, false => 42u8 } as u16;
}
