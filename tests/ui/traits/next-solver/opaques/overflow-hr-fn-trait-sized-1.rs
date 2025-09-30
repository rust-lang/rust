//@ ignore-compare-mode-next-solver
//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#220. Builtin `Fn`-trait
// candidates required `for<'latebound> Output<'latebound>: Sized` which ended
// up resulting in overflow if the return type is an opaque in the defining scope.
//
// We now eagerly instantiate the binder of the function definition which avoids
// that overflow by relating the lifetime of the opaque to something from the
// input.
fn flat_map<T, F, I, G>(_: F, _: G)
where
    F: FnOnce(T) -> I,
    I: Iterator,
    G: Fn(<I as Iterator>::Item) -> usize,
{
}

fn rarw<'a>(_: &'a ()) -> impl Iterator<Item = &'a str> {
    flat_map(rarw, |x| x.len());
    std::iter::empty()
}

fn main() {}
