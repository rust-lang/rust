// Regression test for an issue discovered in https://github.com/rust-lang/rust/pull/143193/files and rediscovered in https://github.com/rust-lang/rust/issues/147254#event-20049906781
// Malformed #[link] attribute was supposed to be deny-by-default report-in-deps FCW,
// but accidentally was landed as a hard error.
//
// revision `default_fcw` tests that with `ill_formed_attribute_input` (the default) denied,
// the attribute produces an FCW
// revision `allowed` tests that with `ill_formed_attribute_input` allowed the test passes

//@ revisions: default_fcw allowed
//@[allowed] check-pass

#[cfg_attr(allowed, allow(ill_formed_attribute_input))]

#[link="dl"]
//[default_fcw]~^ ERROR valid forms for the attribute are
//[default_fcw]~| WARN previously accepted
extern "C" { }

fn main() {}
