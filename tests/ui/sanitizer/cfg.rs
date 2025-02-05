// Verifies that when compiling with -Csanitize=option,
// the `#[cfg(sanitize = "option")]` attribute is configured.

//@ check-pass
//@ revisions: address cfi kcfi leak memory thread
//@compile-flags: -Ctarget-feature=-crt-static
//@[address]needs-sanitizer-address
//@[address]compile-flags: -Zunstable-options -Csanitize=address
//@[cfi]needs-sanitizer-cfi
//@[cfi]compile-flags:     -Zunstable-options -Csanitize=cfi
//@[cfi]compile-flags:     -Clto -Ccodegen-units=1
//@[kcfi]needs-llvm-components: x86
//@[kcfi]compile-flags:    -Zunstable-options -Csanitize=kcfi
//@[kcfi]compile-flags:    --target x86_64-unknown-none -C panic=abort
//@[leak]needs-sanitizer-leak
//@[leak]compile-flags:    -Zunstable-options -Csanitize=leak
//@[memory]needs-sanitizer-memory
//@[memory]compile-flags:  -Zunstable-options -Csanitize=memory
//@[thread]needs-sanitizer-thread
//@[thread]compile-flags:  -Zunstable-options -Csanitize=thread

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
