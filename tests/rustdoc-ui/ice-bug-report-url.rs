// compile-flags: -Ztreat-err-as-bug
// failure-status: 101
// error-pattern: aborting due to
// error-pattern: we would appreciate a bug report: https://github.com/rust-lang/rust/issues/new?labels=C-bug%2C+I-ICE%2C+T-rustdoc&template=ice.md

// normalize-stderr-test "note: compiler flags.*\n\n" -> ""
// normalize-stderr-test "note: rustc.*running on.*" -> "note: rustc {version} running on {platform}"
// normalize-stderr-test "thread.*panicked at .*, compiler.*" -> "thread panicked at 'aborting due to `-Z treat-err-as-bug`'"
// normalize-stderr-test "\s*\d{1,}: .*\n" -> ""
// normalize-stderr-test "\s at .*\n" -> ""
// normalize-stderr-test ".*note: Some details are omitted.*\n" -> ""

fn wrong()
//~^ ERROR expected one of
