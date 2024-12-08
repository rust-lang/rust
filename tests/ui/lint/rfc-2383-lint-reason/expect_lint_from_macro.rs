//@ check-pass

#![warn(unused_variables)]

macro_rules! trigger_unused_variables_macro {
    () => {
        let x = 0;
        //~^ WARNING unused variable: `x` [unused_variables]
        //~| WARNING unused variable: `x` [unused_variables]
    };
}

pub fn check_macro() {
    // This should trigger the `unused_variables` from inside the macro
    trigger_unused_variables_macro!();
}

// This should be fulfilled by the macro
#[expect(unused_variables)]
pub fn check_expect_on_item() {
    trigger_unused_variables_macro!();
}

pub fn check_expect_on_macro() {
    // This should be fulfilled by the macro
    #[expect(unused_variables)]
    trigger_unused_variables_macro!();

    // FIXME: Lint attributes currently don't work directly on macros, and
    // therefore also doesn't work for the new `expect` attribute. This bug
    // is being tracked in rust#87391. The test will until then produce two
    // warnings about the unused variable x.
    //
    // The expectation is still marked as fulfilled. I'm not totally why but
    // my guess is that this will remain working when rust#87391 has been fixed.
}

fn main() {

}
