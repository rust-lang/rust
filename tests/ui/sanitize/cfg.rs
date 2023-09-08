// Verifies that when compiling with -Zsanitizer=option,
// the `#[cfg(sanitize = "option")]` attribute is configured.

// needs-sanitizer-support
// check-pass
// revisions: address cfi kcfi leak memory thread
//[address]needs-sanitizer-address
//[address]compile-flags: -Zsanitizer=address --cfg address
//[cfi]needs-sanitizer-cfi
//[cfi]compile-flags:     -Zsanitizer=cfi     --cfg cfi -Clto
//[kcfi]needs-sanitizer-kcfi
//[kcfi]compile-flags:    -Zsanitizer=kcfi    --cfg kcfi
//[leak]needs-sanitizer-leak
//[leak]compile-flags:    -Zsanitizer=leak    --cfg leak
//[memory]needs-sanitizer-memory
//[memory]compile-flags:  -Zsanitizer=memory  --cfg memory
//[thread]needs-sanitizer-thread
//[thread]compile-flags:  -Zsanitizer=thread  --cfg thread

#![feature(cfg_sanitize)]

#[cfg(all(sanitize = "address", address))]
fn main() {}

#[cfg(all(sanitize = "cfi", cfi))]
fn main() {}

#[cfg(all(sanitize = "kcfi", kcfi))]
fn main() {}

#[cfg(all(sanitize = "leak", leak))]
fn main() {}

#[cfg(all(sanitize = "memory", memory))]
fn main() {}

#[cfg(all(sanitize = "thread", thread))]
fn main() {}
