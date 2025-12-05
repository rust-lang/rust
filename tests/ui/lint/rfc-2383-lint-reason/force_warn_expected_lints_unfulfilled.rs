//@ compile-flags: --force-warn while_true
//@ compile-flags: --force-warn unused_variables
//@ compile-flags: --force-warn unused_mut
//@ check-pass

fn expect_early_pass_lint(terminate: bool) {
    #[expect(while_true)]
    //~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
    //~| NOTE `#[warn(unfulfilled_lint_expectations)]` on by default
    while !terminate {
        println!("Do you know what a spin lock is?")
    }
}

#[expect(unused_variables, reason="<this should fail and display this reason>")]
//~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
//~| NOTE <this should fail and display this reason>
fn check_specific_lint() {
    let _x = 2;
}

#[expect(unused)]
//~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
fn check_multiple_lints_with_lint_group() {
    let fox_name = "Sir Nibbles";

    let what_does_the_fox_say = "*ding* *deng* *dung*";

    println!("The fox says: {what_does_the_fox_say}");
    println!("~ {fox_name}")
}


#[expect(unused)]
//~^ WARNING this lint expectation is unfulfilled [unfulfilled_lint_expectations]
fn check_overridden_expectation_lint_level() {
    #[allow(unused_variables)]
    let this_should_not_fulfill_the_expectation = "maybe";
    //~^ WARNING unused variable: `this_should_not_fulfill_the_expectation` [unused_variables]
    //~| NOTE requested on the command line with `--force-warn unused-variables`
    //~| HELP if this is intentional, prefix it with an underscore
}

fn main() {
    check_multiple_lints_with_lint_group();
    check_overridden_expectation_lint_level();
}
