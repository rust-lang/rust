// Check that we check that default associated types satisfy the required
// bounds on them.

#![feature(specialization)]
//~^ WARNING `specialization` is incomplete

trait X {
    type U: Clone;
    fn unsafe_clone(&self, x: Option<&Self::U>) {
        x.cloned();
    }
}

// We cannot normalize `<T as X>::U` to `str` here, because the default could
// be overridden. The error here must therefore be found by a method other than
// normalization.
impl<T> X for T {
    default type U = str;
    //~^ ERROR trait `Clone` is not implemented for `str`
}

pub fn main() {
    1.unsafe_clone(None);
}
