// check-pass

#![feature(lint_reasons)]

#[expect(unused_mut, reason = "This should trigger because `unused_mut` was allow")]
//~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectation]
//~| NOTE `#[warn(unfulfilled_lint_expectation)]` on by default
//~| NOTE This should trigger because `unused_mut` was allow
mod foo {
  fn bar() {
    #[allow(unused_mut, reason = "v is unused")]
    let mut v = 0;
  }
}

#[expect(unused_mut, reason = "This should trigger because `unused_mut` is no longer expected")]
//~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectation]
//~| NOTE This should trigger because `unused_mut` is no longer expected
mod oof {
    #[warn(unused_mut, reason = "We no longer expect it in this scope")]
    //~^ NOTE the lint level is defined here
    fn bar() {
        let mut v = 0;
        //~^ WARNING variable does not need to be mutable [unused_mut]
        //~| NOTE We no longer expect it in this scope
    }
}

fn main() {}
