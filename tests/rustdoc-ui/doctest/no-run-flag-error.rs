// test the behavior of the --no-run flag without the --test flag

//@ compile-flags:-Z unstable-options --no-run --test-args=--test-threads=1

pub fn f() {}

//~? ERROR the `--test` flag must be passed to enable `--no-run`
