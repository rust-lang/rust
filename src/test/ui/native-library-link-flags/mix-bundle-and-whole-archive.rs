// Mixing +bundle and +whole-archive is not allowed

// compile-flags: -l static:+bundle,+whole-archive=mylib -Zunstable-options --crate-type rlib
// build-fail
// error-pattern: the linking modifiers `+bundle` and `+whole-archive` are not compatible with each other when generating rlibs

fn main() { }
