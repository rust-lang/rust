// Regression test for <https://github.com/rust-lang/rust/issues/141844>.

fn main() {
  // The following expression gets desugared into something like:
  // ```
  // let (lhs,) = x; (let x = 1) = lhs;
  // ```
  // This used to ICE since we haven't yet declared the type for `x` when
  // checking the first desugared statement, whose RHS resolved to `x` since
  // in the AST, the `let` expression was visited first.
  (let x = 1,) = x;
  //~^ ERROR expected expression, found `let` statement
}
