//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[current] check-pass

// Test from trait-system-refactor-initiative#176.
//
// Normalizing `<Vec<T> as IntoIterator>::IntoIter` has two candidates
// inside of the function:
// - `impl<T> IntoIterator for Vec<T>` which trivially applies
// - `impl<I: Iterator> IntoIterator for I`
//   - requires `Vec<T>: Iterator`
//     - where-clause requires `<Vec<T> as IntoIterator>::IntoIter eq Vec<T>`
//       - normalize `<Vec<T> as IntoIterator>::IntoIter` again, cycle
//
// The blanket impl is unfortunately also a productive cycle, so we have to
// break this code, see trait-system-refactor-initiative#273

fn test<T>()
where
    <Vec<T> as IntoIterator>::IntoIter: Iterator,
    //[next]~^ ERROR type annotations needed: cannot satisfy `<Vec<T> as IntoIterator>::IntoIter: Iterator`
{
}

fn main() {}
