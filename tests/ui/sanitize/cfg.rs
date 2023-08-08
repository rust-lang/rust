// Verifies that when compiling with -Zsanitizer=option,
// the `#[cfg(sanitize = "option")]` attribute is configured.

// needs-sanitizer-support
// needs-sanitizer-address
// needs-sanitizer-cfi
// needs-sanitizer-kcfi
// needs-sanitizer-leak
// needs-sanitizer-memory
// needs-sanitizer-thread
// check-pass
// revisions: address leak memory thread
//[address]compile-flags: -Zsanitizer=address --cfg address
//[cfi]compile-flags:     -Zsanitizer=cfi     --cfg cfi
//[kcfi]compile-flags:    -Zsanitizer=kcfi    --cfg kcfi
//[leak]compile-flags:    -Zsanitizer=leak    --cfg leak
//[memory]compile-flags:  -Zsanitizer=memory  --cfg memory
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
