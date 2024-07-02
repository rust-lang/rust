//@ run-pass
//@ compile-flags: --cfg test -Aunexpected_builtin_cfgs

// Make sure `--cfg test` does not inject test harness

#[test]
fn test() { panic!(); }

fn main() {}
