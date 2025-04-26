//@ compile-flags: --crate-type=lib
//@ check-pass

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
