// Verifies that when compiling with -Zsanitizer=option,
// the `#[cfg(sanitize = "option")]` attribute is configured.

// needs-sanitizer-support
// only-linux
// only-x86_64
// check-pass
// revisions: address leak memory thread
//[address]compile-flags: -Zsanitizer=address --cfg address
//[leak]compile-flags:    -Zsanitizer=leak    --cfg leak
//[memory]compile-flags:  -Zsanitizer=memory  --cfg memory
//[thread]compile-flags:  -Zsanitizer=thread  --cfg thread

#![feature(cfg_sanitize)]

#[cfg(all(sanitize = "address", address))]
fn main() {}

#[cfg(all(sanitize = "leak", leak))]
fn main() {}

#[cfg(all(sanitize = "memory", memory))]
fn main() {}

#[cfg(all(sanitize = "thread", thread))]
fn main() {}
