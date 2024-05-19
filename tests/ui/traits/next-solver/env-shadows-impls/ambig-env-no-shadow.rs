//@ compile-flags: -Znext-solver
//@ check-pass

// If a trait goal is proven using the environment, we discard
// impl candidates when normalizing. However, in this example
// the env candidates start as ambiguous and end up not applying,
// so normalization should succeed later on.

trait Trait<T>: Sized {
    type Assoc: From<Self>;
}

impl<T, U> Trait<U> for T {
    type Assoc = T;
}

fn mk_assoc<T: Trait<U>, U>(t: T, _: U) -> <T as Trait<U>>::Assoc {
    t.into()
}

fn generic<T>(t: T) -> T
where
    T: Trait<u32>,
    T: Trait<i16>,
{
    let u = Default::default();

    // at this point we have 2 ambig env candidates
    let ret: T = mk_assoc(t, u);

    // now both env candidates don't apply, so we're now able to
    // normalize using this impl candidates. For this to work
    // the normalizes-to must have remained ambiguous above.
    let _: u8 = u;
    ret
}

fn main() {
    assert_eq!(generic(1), 1);
}
