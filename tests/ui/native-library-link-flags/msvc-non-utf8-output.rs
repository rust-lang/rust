// build-fail
//@compile-flags:-C link-arg=⦺ⅈ⽯⭏⽽◃⡽⚞
//@only-target-msvc
// normalize-stderr-test "(?:.|\n)*(⦺ⅈ⽯⭏⽽◃⡽⚞)(?:.|\n)*" -> "$1"
pub fn main() {}
