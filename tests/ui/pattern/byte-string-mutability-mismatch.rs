//! Byte string literal patterns use the mutability of the literal, rather than the mutability of
//! the pattern's scrutinee. Since byte string literals are always shared references, it's a
//! mismatch to use a byte string literal pattern to match on a mutable array or slice reference.

//@ dont-require-annotations: NOTE

fn main() {
    let mut val = [97u8, 10u8];
    match &mut val {
        b"a\n" => {},
        //~^ ERROR mismatched types
        //~| NOTE types differ in mutability
        _ => {},
    }
    match &mut val[..] {
         b"a\n" => {},
        //~^ ERROR mismatched types
        //~| NOTE types differ in mutability
         _ => {},
    }
}
