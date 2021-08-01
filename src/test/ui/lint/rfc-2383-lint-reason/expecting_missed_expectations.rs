// check-pass

#![feature(lint_reasons)]

#[expect(unfulfilled_lint_expectations, reason = "this should catch the nested expect")]
mod foo {
  #[expect(unfulfilled_lint_expectations, reason = "issuing a lint and getting caught above")]
  fn bar() {
    #[expect(unused_mut, reason = "v is unused")]
    let mut v = 0;
  }
}

// make sure it doesn't catch itself
#[expect(unfulfilled_lint_expectations, reason = "this should issue a warning")]
//~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
//~| NOTE `#[warn(unfulfilled_lint_expectations)]` on by default
//~| NOTE this should issue a warning
mod oof {}

fn main() {}
