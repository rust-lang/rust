use std::ops::Deref;

pub fn func<'a>(_x: impl Clone + Into<Vec<u8>> + 'a) {}

pub fn func2<T>(
    _x: impl Deref<Target = Option<T>> + Iterator<Item = T>,
    _y: impl Iterator<Item = u8>,
) {}

pub fn func3(_x: impl Iterator<Item = impl Iterator<Item = u8>> + Clone) {}

pub fn func4<T: Iterator<Item = impl Clone>>(_x: T) {}

pub fn func5(
    _f: impl for<'any> Fn(&'any str, &'any str) -> bool + for<'r> Other<T<'r> = ()>,
    _a: impl for<'beta, 'alpha, '_gamma> Auxiliary<'alpha, Item<'beta> = fn(&'beta ())>,
) {}

pub trait Other {
    type T<'dependency>;
}

pub trait Auxiliary<'arena> {
    type Item<'input>;
}

pub struct Foo;

impl Foo {
    pub fn method<'a>(_x: impl Clone + Into<Vec<u8>> + 'a) {}
}

// Regression test for issue #113015, subitem (1) / bevyengine/bevy#8898.
// Check that we pick up on the return type of cross-crate opaque `Fn`s and
// `FnMut`s (`FnOnce` already used to work fine).
pub fn rpit_fn() -> impl Fn() -> bool {
    || false
}

pub fn rpit_fn_mut() -> impl for<'a> FnMut(&'a str) -> &'a str {
    |source| source
}

// FIXME(fmease): Add more tests that demonstrate the importance of checking the
// generic args of supertrait bounds.
