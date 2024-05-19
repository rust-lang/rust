//@ run-pass
//@ compile-flags: --cfg test

// Make sure `--cfg test` does not inject test harness

#[test]
fn test() { panic!(); }

fn main() {}
