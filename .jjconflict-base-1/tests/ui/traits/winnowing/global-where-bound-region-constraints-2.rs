//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#172.
//
// In this test the global where-bound simply constrains the
// object lifetime bound to 'static while the builtin impl
// ends up also emitting a `dyn Any: 'static` type outlives
// constraint. This previously resulted in ambiguity. We now
// always prefer the impl.

pub trait Any: 'static {}

pub trait Downcast<T>: Any
where
    T: Any,
{
}

// elided object lifetime: `dyn Any + 'static`
impl dyn Any {
    pub fn is<T>(&self)
    where
        T: Any,
        // elaboration adds global where-clause `dyn Any + 'static: Any`
        Self: Downcast<T>,
    {
    }
}

fn main() {}
