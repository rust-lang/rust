//@compile-flags:-l bar=foo
// ignore-tidy-linelength
//@error-in-other-file: unknown library kind `bar`, expected one of: static, dylib, framework, link-arg

fn main() {
}
