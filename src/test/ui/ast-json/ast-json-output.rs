// Check that AST json printing works.

// check-pass
// compile-flags: -Zast-json-noexpand
// normalize-stdout-test ":\d+" -> ":0"

// Only include a single item to reduce how often the test output needs
// updating.
extern crate core;
