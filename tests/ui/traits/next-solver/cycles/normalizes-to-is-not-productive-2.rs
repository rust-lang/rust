//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#176.
//
// Normalizing `<Vec<T> as IntoIterator>::IntoIter` has two candidates
// inside of the function:
// - `impl<T> IntoIterator for Vec<T>` which trivially applies
// - `impl<I: Iterator> IntoIterator for I`
//   - requires `Vec<T>: Iterator`
//     - where-clause requires `<Vec<T> as IntoIterator>::IntoIter eq Vec<T>`
//       - normalize `<Vec<T> as IntoIterator>::IntoIter` again, cycle
//
// We need to treat this cycle as an error to be able to use the actual impl.

fn test<T>()
where
    <Vec<T> as IntoIterator>::IntoIter: Iterator,
{
}

fn main() {}
