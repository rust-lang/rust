// This test verifies that unstable options like `-Zcrate-attr` are respected when `--test` is
// passed.
//
// <https://github.com/rust-lang/rust/issues/147276>
//
// NOTE: If any of these command line arguments or features get stabilized, please replace with
// another unstable one.

//@ revisions: normal crate_attr
//@ compile-flags: --test
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@[crate_attr] check-pass
//@[crate_attr] compile-flags: -Zcrate-attr=feature(used_with_arg)

#[used(linker)]
//[normal]~^ ERROR `#[used(linker)]` is currently unstable
static REPRO: isize = 1;
