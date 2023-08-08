// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// needs-unwind

#![crate_type = "lib"]
pub trait Factory<T> {
    type Item;
}

pub struct IntFactory;

impl<T> Factory<T> for IntFactory {
    type Item = usize;
}

// EMIT_MIR caller_with_trivial_bound.foo.Inline.diff
pub fn foo<T>()
where
    IntFactory: Factory<T>,
{
    let mut x: <IntFactory as Factory<T>>::Item = bar::<T>();
}

#[inline(always)]
pub fn bar<T>() -> <IntFactory as Factory<T>>::Item {
    0usize
}
