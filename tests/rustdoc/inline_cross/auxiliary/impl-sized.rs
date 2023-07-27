use std::fmt::Debug;

pub fn sized(x: impl Sized) -> impl Sized {
    x
}

pub fn sized_outlives<'a>(x: impl Sized + 'a) -> impl Sized + 'a {
    x
}

pub fn maybe_sized(x: &impl ?Sized) -> &impl ?Sized {
    x
}

pub fn debug_maybe_sized(x: &(impl Debug + ?Sized)) -> &(impl Debug + ?Sized) {
    x
}

pub fn maybe_sized_outlives<'t>(x: &(impl ?Sized + 't)) -> &(impl ?Sized + 't) {
    x
}
