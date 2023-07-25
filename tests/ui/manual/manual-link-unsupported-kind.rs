//@compile-flags:-l raw-dylib=foo
// ignore-tidy-linelength
//@error-pattern: unknown library kind `raw-dylib`, expected one of: static, dylib, framework, link-arg

fn main() {
}
