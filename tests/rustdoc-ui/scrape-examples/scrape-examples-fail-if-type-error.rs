//@ check-fail
//@ compile-flags: -Z unstable-options --scrape-examples-output-path {{build-base}}/t.calls --scrape-examples-target-crate foobar

pub fn foo() {
  INVALID_FUNC();
  //~^ ERROR could not resolve path
}

//~? ERROR Compilation failed, aborting rustdoc
