// build-fail
// aux-build:lto-duplicate-symbols1.rs
// aux-build:lto-duplicate-symbols2.rs
// error-pattern:Linking globals named 'foo': symbol multiply defined!
// compile-flags: -C lto
// no-prefer-dynamic
// normalize-stderr-test: "lto-duplicate-symbols2\.lto_duplicate_symbols2\.[0-9a-zA-Z]+-cgu" -> "lto-duplicate-symbols2.lto_duplicate_symbols2.HASH-cgu"
extern crate lto_duplicate_symbols1;
extern crate lto_duplicate_symbols2;

fn main() {}
