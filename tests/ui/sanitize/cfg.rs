// Verifies that when compiling with -Zsanitizer=option,
// the `#[cfg(sanitize = "option")]` attribute is configured.

//@ check-pass
//@ revisions: address cfi kcfi leak memory thread
//@compile-flags: -Ctarget-feature=-crt-static
//@[address]needs-sanitizer-address
//@[address]compile-flags: -Zsanitizer=address --cfg address
//@[cfi]needs-sanitizer-cfi
//@[cfi]compile-flags:     -Zsanitizer=cfi     --cfg cfi
//@[cfi]compile-flags:     -Clto -Ccodegen-units=1
//@[kcfi]needs-llvm-components: x86
//@[kcfi]compile-flags:    -Zsanitizer=kcfi    --cfg kcfi --target x86_64-unknown-none
//@[leak]needs-sanitizer-leak
//@[leak]compile-flags:    -Zsanitizer=leak    --cfg leak
//@[memory]needs-sanitizer-memory
//@[memory]compile-flags:  -Zsanitizer=memory  --cfg memory
//@[thread]needs-sanitizer-thread
//@[thread]compile-flags:  -Zsanitizer=thread  --cfg thread

#![feature(cfg_sanitize, no_core, lang_items)]
#![crate_type="lib"]
#![no_core]

#[lang="sized"]
trait Sized { }
#[lang="copy"]
trait Copy { }

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

pub fn check() {
    main();
}
