//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
//[current]~^^^^ ERROR type annotations needed: cannot satisfy `Self: Gen<'source>`

// This previously failed with ambiguity in the old solver and
// was added as a diagnostics test in #105285. The new solver
// is now able to uniquely chose one of the `ParamEnv` candidates
// as it does not result in any region constraints, causing this
// test to pass.

pub trait Gen<'source> {
    type Output;

    fn gen<T>(&self) -> T
    where
        Self: for<'s> Gen<'s, Output = T>;
}

fn main() {}
