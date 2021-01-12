// compile-flags: --test -Z unstable-options --test-builder true --runtool true

/// ```
/// This does not compile, but specifying a custom --test-builder should let this pass anyway
/// `true` does not generate an output file to run, so we also specify it as a runtool
/// ```
pub struct Foo;
