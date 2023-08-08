// only-macos
// compile-flags: --print deployment-target
// normalize-stdout-test: "\d+\." -> "$$CURRENT_MAJOR_VERSION."
// normalize-stdout-test: "\d+" -> "$$CURRENT_MINOR_VERSION"
// check-pass

fn main() {}
