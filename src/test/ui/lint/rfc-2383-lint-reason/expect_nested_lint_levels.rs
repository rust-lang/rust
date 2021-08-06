// check-pass
// ignore-tidy-linelength

#![feature(lint_reasons)]
#![warn(unused_mut)]

#[expect(
    unused_mut,
    reason = "this `expect` is overridden by a `allow` attribute before the `unused_mut` lint is triggered"
)]
//~^^^^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
//~| NOTE this `expect` is overridden by a `allow` attribute before the `unused_mut` lint is triggered
mod foo {
    fn bar() {
        #[allow(
            unused_mut,
            reason = "this overrides the previous `expect` lint level and allows the `unused_mut` lint here"
        )]
        let mut v = 0;
    }
}

#[expect(
    unused_mut,
    reason = "this `expect` is overridden by a `warn` attribute before the `unused_mut` lint is triggered"
)]
//~^^^^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
//~| NOTE `#[warn(unfulfilled_lint_expectations)]` on by default
//~| NOTE this `expect` is overridden by a `warn` attribute before the `unused_mut` lint is triggered
mod oof {
    #[warn(
        unused_mut,
        //~^ NOTE the lint level is defined here
        reason = "this overrides the previous `expect` lint level and warns about the `unused_mut` lint here"
    )]
    fn bar() {
        let mut v = 0;
        //~^ WARNING variable does not need to be mutable [unused_mut]
        //~| NOTE this overrides the previous `expect` lint level and warns about the `unused_mut` lint here
    }
}

fn main() {}
