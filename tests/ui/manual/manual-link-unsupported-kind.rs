//@compile-flags:-l raw-dylib=foo
// ignore-tidy-linelength
//@error-in-other-file: unknown library kind `raw-dylib`, expected one of: static, dylib, framework, link-arg

fn main() {
}
