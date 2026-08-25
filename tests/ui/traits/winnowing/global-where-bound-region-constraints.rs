//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#172.
//
// The next-generation trait solver previously simply tried
// to merge the global where-bounds with the impl candidates.
// This caused ambiguity in case the where-bound had stricter
// region requirements than the impl.

trait Trait {}
struct Foo<'a, 'b>(&'a (), &'b ());
impl<'a> Trait for Foo<'a, 'static> {}

fn impls_trait<T: Trait>() {}
fn foo()
where
    Foo<'static, 'static>: Trait,
{
    // impl requires `'1 to be 'static
    // global where-bound requires both '0 and '1 to be 'static
    //
    // we always prefer the impl here.
    impls_trait::<Foo<'_, '_>>();
}

fn main() {}
