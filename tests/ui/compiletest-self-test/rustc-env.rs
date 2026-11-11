// Compiletest meta test checking that rustc-env and unset-rustc-env directives
// can be used to configure environment for rustc.
//
//@ run-pass
//@ aux-build:env.rs
//@ rustc-env:COMPILETEST_FOO=foo
//
// An environment variable that is likely to be set, but should be safe to unset.
//@ unset-rustc-env:PWD

extern crate env;

fn main() {
    assert_eq!(env!("COMPILETEST_FOO"), "foo");
    assert_eq!(option_env!("COMPILETEST_BAR"), None);
    assert_eq!(option_env!("PWD"), None);
    env::test();
}
