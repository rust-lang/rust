// Check that AST json printing works.
#![crate_type = "lib"]

// check-pass
// compile-flags: -Zast-json
// normalize-stdout-test ":\d+" -> ":0"

// Only include a single item to reduce how often the test output needs
// updating.
extern crate core;
