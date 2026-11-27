// ICE Failed to normalize closure with TAIT
// issue: rust-lang/rust#109020
//@ check-pass

#![feature(type_alias_impl_trait)]

use std::marker::PhantomData;

type WithEmplacableForFn<'a> = impl EmplacableFn + 'a;

#[define_opaque(WithEmplacableForFn)]
fn _constrain(_: &mut ()) -> WithEmplacableForFn<'_> {
    ()
}

fn with_emplacable_for<'a, F, R>(mut f: F) -> R
where
    F: for<'b> FnMut(Emplacable<WithEmplacableForFn<'b>>) -> R,
{
    fn with_emplacable_for_inner<'a, R>(
        _: &'a (),
        _: &mut dyn FnMut(Emplacable<WithEmplacableForFn<'a>>) -> R,
    ) -> R {
        loop {}
    }

    with_emplacable_for_inner(&(), &mut f)
}

trait EmplacableFn {}

impl EmplacableFn for () {}

struct Emplacable<F>
where
    F: EmplacableFn,
{
    phantom: PhantomData<F>,
}

fn main() {
    with_emplacable_for(|_| {});
}
