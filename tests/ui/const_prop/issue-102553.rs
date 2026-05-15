//@ compile-flags: --crate-type=lib -Zmir-opt-level=3
//@ build-pass

pub trait Widget<E> {
    fn boxed<'w>(self) -> Box<dyn WidgetDyn<E> + 'w>
    where
        Self: Sized + 'w;
}

pub trait WidgetDyn<E> {}

impl<T, E> WidgetDyn<E> for T where T: Widget<E> {}

impl<E> Widget<E> for dyn WidgetDyn<E> + '_ {
    fn boxed<'w>(self) -> Box<dyn WidgetDyn<E> + 'w>
    where
        Self: Sized + 'w,
    {
        // Even though this is illegal to const evaluate, this should never
        // trigger an ICE because it can never be called from actual code
        // (due to the trivially false where-clause predicate).
        Box::new(self)
    }
}

// Regression test for https://github.com/rust-lang/rust/issues/156051.
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
