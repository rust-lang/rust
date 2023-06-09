// This must fail coherence.
//
// Getting this to pass was fairly difficult, so here's an explanation
// of what's happening:
//
// Normalizing projections currently tries to replace them with inference variables
// while emitting a nested `Projection` obligation. This cannot be done if the projection
// has bound variables which is the case here.
//
// So the projections stay until after normalization. When unifying two projections we
// currently treat them as if they are injective, so we **incorrectly** unify their
// substs. This means that coherence for the two impls ends up unifying `?T` and `?U`
// as it tries to unify `<?T as WithAssoc1<'a>>::Assoc` with `<?U as WithAssoc1<'a>>::Assoc`.
//
// `impl1` therefore has the projection `<?T as WithAssoc2<'a>>::Assoc` and we have the
// assumption `?T: for<'a> WithAssoc2<'a, Assoc = i32>` in the `param_env`, so we normalize
// that to `i32`. We then try to unify `i32` from `impl1` with `u32` from `impl2` which fails,
// causing coherence to consider these two impls distinct.

// compile-flags: -Ztrait-solver=next
pub trait Trait<T> {}

pub trait WithAssoc1<'a> {
    type Assoc;
}
pub trait WithAssoc2<'a> {
    type Assoc;
}

// impl 1
impl<T, U> Trait<for<'a> fn(<T as WithAssoc1<'a>>::Assoc, <U as WithAssoc2<'a>>::Assoc)> for (T, U)
where
    T: for<'a> WithAssoc1<'a> + for<'a> WithAssoc2<'a, Assoc = i32>,
    U: for<'a> WithAssoc2<'a>,
{
}

// impl 2
impl<T, U> Trait<for<'a> fn(<U as WithAssoc1<'a>>::Assoc, u32)> for (T, U) where
    U: for<'a> WithAssoc1<'a> //~^ ERROR conflicting implementations of trait
{
}

fn main() {}
