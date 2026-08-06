// Regression test for https://github.com/rust-lang/rust/issues/156051.

//@ compile-flags: --crate-type=lib -Zmir-opt-level=3
//@ build-pass

trait WidgetSize<E> {
    fn size<'w>(self) -> usize
    where
        Self: Sized + 'w;
}

trait WidgetSizeDyn<E> {}

impl<E> WidgetSize<E> for dyn WidgetSizeDyn<E> + '_ {
    fn size<'w>(self) -> usize
    where
        Self: Sized + 'w,
    {
        core::mem::size_of::<Self>()
    }
}
