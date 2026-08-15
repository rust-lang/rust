//! This test demonstrates how `LayoutError::NormalizationFailure` can happen and why
//! it is necessary.
//!
//! This code does not cause an immediate normalization error in typeck, because we
//! don't reveal the hidden type returned by `opaque<T>` in the analysis typing mode.
//! Instead, `<{opaque} as Project2>::Assoc2` is a *rigid projection*, because we know
//! that `{opaque}: Project2` holds, due to the opaque type's `impl Project2` bound,
//! but cannot normalize `<{opaque} as Project2>::Assoc2` any further.
//!
//! However, in the post-analysis typing mode, which is used for the layout computation,
//! the opaque's hidden type is revealed to be `PhantomData<T>`, and now we fail to
//! normalize `<PhantomData<T> as Project2>::Assoc2` if there is a `T: Project1` bound
//! in the param env! This happens, because `PhantomData<T>: Project2` only holds if
//! `<T as Project1>::Assoc1 == ()` holds. This would usually be satisfied by the
//! blanket `impl<T> Project1 for T`, but due to the `T: Project1` bound we do not
//! normalize `<T as Project1>::Assoc1` via the impl and treat it as rigid instead.
//! Therefore, `PhantomData<T>: Project2` does NOT hold and normalizing
//! `<PhantomData<T> as Project2>::Assoc2` fails.
//!
//! Note that this layout error can only happen when computing the layout in a generic
//! context, which is not required for codegen, but may happen for lints, MIR optimizations,
//! and the transmute check.

use std::marker::PhantomData;

trait Project1 {
    type Assoc1;
}

impl<T> Project1 for T {
    type Assoc1 = ();
}

trait Project2 {
    type Assoc2;
    fn get(self) -> Self::Assoc2;
}

impl<T: Project1<Assoc1 = ()>> Project2 for PhantomData<T> {
    type Assoc2 = ();
    fn get(self) -> Self::Assoc2 {}
}

fn opaque<T>() -> impl Project2 {
    PhantomData::<T>
}

fn check<T: Project1>() {
    unsafe {
        std::mem::transmute::<_, ()>(opaque::<T>().get());
        //~^ ERROR: cannot transmute
        //~| NOTE: source type: `{type error}` (the type has an unknown layout)
        //~| NOTE: target type: `()` (0 bits)
    }
}

fn main() {}
