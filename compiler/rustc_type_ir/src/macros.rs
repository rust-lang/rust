/// Used for types that are `Copy` and which **do not care arena
/// allocated data** (i.e., don't need to be folded).
macro_rules! TrivialTypeTraversalImpls {
    ($($ty:ty,)+) => {
        $(
            impl<I: $crate::Interner> $crate::fold::TypeFoldable<I> for $ty {
                fn try_fold_with<F: $crate::fold::FallibleTypeFolder<I>>(
                    self,
                    _: &mut F,
                ) -> ::std::result::Result<Self, F::Error> {
                    Ok(self)
                }

                #[inline]
                fn fold_with<F: $crate::fold::TypeFolder<I>>(
                    self,
                    _: &mut F,
                ) -> Self {
                    self
                }
            }

            impl<I: $crate::Interner> $crate::visit::TypeVisitable<I> for $ty {
                #[inline]
                fn visit_with<F: $crate::visit::TypeVisitor<I>>(
                    &self,
                    _: &mut F)
                    -> ::std::ops::ControlFlow<F::BreakTy>
                {
                    ::std::ops::ControlFlow::Continue(())
                }
            }
        )+
    };
}

///////////////////////////////////////////////////////////////////////////
// Atomic structs
//
// For things that don't carry any arena-allocated data (and are
// copy...), just add them to this list.

TrivialTypeTraversalImpls! {
    (),
    bool,
    usize,
    crate::AliasRelationDirection,
    crate::UniverseIndex,
}

#[macro_export]
macro_rules! noop_if_trivially_traversable {
    ($val:tt.try_fold_with::<$interner:ty>($folder:expr)) => {{
        use $crate::fold::SpecTypeFoldable as _;
        $crate::noop_if_trivially_traversable!($val.spec_try_fold_with::<$interner>($folder))
    }};
    ($val:tt.visit_with::<$interner:ty>($visitor:expr)) => {{
        use $crate::visit::SpecTypeVisitable as _;
        $crate::noop_if_trivially_traversable!($val.spec_visit_with::<$interner>($visitor))
    }};
    ($val:tt.$method:ident::<$interner:ty>($traverser:expr)) => {{
        let val = $val;

        #[allow(unreachable_code)]
        let p = 'p: {
            use ::core::marker::PhantomData;

            fn unreachable_phantom_constraint<I, T>(_: T) -> PhantomData<(I, T)> {
                unreachable!()
            }

            break 'p PhantomData;
            unreachable_phantom_constraint::<$interner, _>(val)
        };

        (&&p).$method(val, $traverser)
    }};
}
