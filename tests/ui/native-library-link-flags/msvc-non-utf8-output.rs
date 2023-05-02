// build-fail
// compile-flags:-C link-arg=märchenhaft
// only-msvc
// error-pattern:= note: LINK : fatal error LNK1181:
// normalize-stderr-test "(\s*\|\n)\s*= note: .*\n" -> "$1"
pub fn main() {}
