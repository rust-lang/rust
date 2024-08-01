// ignore-tidy-linelength

#![warn(unused_mut)]

#[expect(
    unused_mut,
    //~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
    //~| NOTE `#[warn(unfulfilled_lint_expectations)]` on by default
    //~| NOTE this `expect` is overridden by a `allow` attribute before the `unused_mut` lint is triggered
    reason = "this `expect` is overridden by a `allow` attribute before the `unused_mut` lint is triggered"
)]
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
    //~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
    //~| NOTE this `expect` is overridden by a `warn` attribute before the `unused_mut` lint is triggered
    reason = "this `expect` is overridden by a `warn` attribute before the `unused_mut` lint is triggered"
)]
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
        //~| HELP remove this `mut`
    }
}

#[expect(unused_variables)]
//~^ WARNING this lint expectation is unfulfilled
#[forbid(unused_variables)]
//~^ NOTE the lint level is defined here
fn check_expect_then_forbid() {
    let this_is_my_function = 3;
    //~^ ERROR unused variable: `this_is_my_function` [unused_variables]
    //~| HELP if this is intentional, prefix it with an underscore
}

fn main() {}
