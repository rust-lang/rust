// check-pass

#![feature(lint_reasons)]

#[expect(unfulfilled_lint_expectation, reason = "this should catch the nested expect")]
mod foo {
  #[expect(unfulfilled_lint_expectation, reason = "issuing a lint and getting caught above")]
  fn bar() {
    #[expect(unused_mut, reason = "v is unused")]
    let mut v = 0;
  }
}

// make sure it doesn't catch itself
#[expect(unfulfilled_lint_expectation, reason = "this should issue a warning")]
//~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectation]
//~| NOTE `#[warn(unfulfilled_lint_expectation)]` on by default
//~| NOTE this should issue a warning
mod oof {}

fn main() {}
