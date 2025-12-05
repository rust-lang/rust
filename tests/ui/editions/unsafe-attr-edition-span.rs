// Tests that the correct span is used to determine the edition of an attribute that was safe to use
// in earlier editions, but has become `unsafe` in later editions.
//
// Determining the correct edition is non-trivial because of macro expansion. For instance,
// the `thread_local!` macro (defined in std and hence using the most recent edition) parses the
// attribute, and then re-emits it. Therefore, the span of the `#` token starting the
// `#[no_mangle]` attribute has std's edition, while the attribute name has the edition of this
// file, which may be different.

//@ revisions: e2015 e2018 e2021 e2024

//@[e2018] edition:2018
//@[e2021] edition:2021
//@[e2024] edition:2024
//
//@[e2015] check-pass
//@[e2018] check-pass
//@[e2021] check-pass
#![crate_type = "lib"]

#[no_mangle] //[e2024]~ ERROR unsafe attribute used without unsafe
static TEST_OUTSIDE: usize = 0;

thread_local! {
    #[no_mangle]//[e2024]~ ERROR unsafe attribute used without unsafe
    static TEST: usize = 0;
}
