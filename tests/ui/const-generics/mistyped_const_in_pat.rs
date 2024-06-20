//! Used to ICE rust-lang/rust#125799 due to `isize` != `()`
//! not being detected early due to the conflicting impls.
//@ only-x86_64

trait Trait<T> {
    type Assoc;
}

impl<T> Trait<T> for Vec<T> {
    type Assoc = ();
}

impl Trait<u8> for Vec<u8> {}
//~^ ERROR: conflicting implementations

const BAR: <Vec<u8> as Trait<u8>>::Assoc = 3;

pub fn main() {
    let x: isize = 3;
    let _ = match x {
        BAR => 2,
        _ => 3,
    };
}
