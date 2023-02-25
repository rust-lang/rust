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
