// Unspecified kind should fail with an error

// compile-flags: -l =mylib
// error-pattern: unknown library kind ``, expected one of dylib, framework, or static

fn main() {}
