// This test verifies that unstable options like `-Zcrate-attr` are respected when `--test` is
// passed.
//
// <https://github.com/rust-lang/rust/issues/143930>
//
// NOTE: If any of these command line arguments or features get stabilized, please replace with
// another unstable one.

//@ check-pass
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ compile-flags: --test -Zcrate-attr=feature(register_tool) -Zcrate-attr=register_tool(rapx)

#[rapx::tag]
fn f() {}
