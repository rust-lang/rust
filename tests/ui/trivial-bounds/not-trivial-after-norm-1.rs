//@[current] check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait With {
    type Assoc;
}
impl<T> With for T {
    type Assoc = T;
}

trait Incomplete<T> {}
impl<T> Incomplete<T> for () {}
fn impls_incomplete<T: Incomplete<U>, U>() {}
fn foo<T>()
where
    u32: With<Assoc = T>,
    // This where-bound is global before normalization
    // and references `T` afterwards. We check whether
    // global where-bounds hold by proving them in an empty
    // `param_env`.
    //
    // Make sure we don't introduce params by normalizing after
    // checking whether the where-bound is global.
    (): Incomplete<<u32 as With>::Assoc>,
{
    impls_incomplete::<(), _>();
    //[next]~^ ERROR type annotations needed
    // FIXME(-Znext-solver): This should match the behavior of the old solver
}

fn main() {}
