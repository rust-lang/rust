//@ compile-flags: --force-warn while_true
//@ compile-flags: --force-warn unused_variables
//@ compile-flags: --force-warn unused_mut
//@ check-pass

fn expect_early_pass_lint() {
    #[expect(while_true)]
    while true {
        //~^ WARNING denote infinite loops with `loop { ... }` [while_true]
        //~| NOTE requested on the command line with `--force-warn while-true`
        //~| HELP use `loop`
        println!("I never stop")
    }
}

#[expect(unused_variables, reason="<this should fail and display this reason>")]
fn check_specific_lint() {
    let x = 2;
    //~^ WARNING unused variable: `x` [unused_variables]
    //~| NOTE requested on the command line with `--force-warn unused-variables`
    //~| HELP if this is intentional, prefix it with an underscore
}

#[expect(unused)]
fn check_multiple_lints_with_lint_group() {
    let fox_name = "Sir Nibbles";
    //~^ WARNING unused variable: `fox_name` [unused_variables]
    //~| HELP if this is intentional, prefix it with an underscore

    let mut what_does_the_fox_say = "*ding* *deng* *dung*";
    //~^ WARNING variable does not need to be mutable [unused_mut]
    //~| NOTE requested on the command line with `--force-warn unused-mut`
    //~| HELP remove this `mut`

    println!("The fox says: {what_does_the_fox_say}");
}

#[allow(unused_variables)]
fn check_expect_overrides_allow_lint_level() {
    #[expect(unused_variables)]
    let this_should_fulfill_the_expectation = "The `#[allow]` has no power here";
    //~^ WARNING unused variable: `this_should_fulfill_the_expectation` [unused_variables]
    //~| HELP if this is intentional, prefix it with an underscore
}

fn main() {}
