//@ check-pass

// Here we check that a `:vis` macro matcher subsititued for the empty visibility
// (`VisibilityKind::Inherited`) is accepted when used before an item in a trait.

fn main() {}

macro_rules! mac_in_trait {
    ($vis:vis MARKER) => {
        $vis fn beta() {}

        $vis const GAMMA: u8;

        $vis type Delta;
    }
}

trait Alpha {
    mac_in_trait!(MARKER);
}

// We also accept visibilities on items in traits syntactically but not semantically.
#[cfg(false)]
trait Foo {
    pub fn bar();
    pub(crate) type baz;
    pub(super) const QUUX: u8;
}
