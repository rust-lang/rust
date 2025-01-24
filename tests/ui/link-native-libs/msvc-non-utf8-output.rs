//@ build-fail
//@ compile-flags:-C link-arg=⦺ⅈ⽯⭏⽽◃⡽⚞
//@ only-msvc
//@ normalize-stderr: "(?:.|\n)*(⦺ⅈ⽯⭏⽽◃⡽⚞)(?:.|\n)*" -> "$1"
pub fn main() {}
