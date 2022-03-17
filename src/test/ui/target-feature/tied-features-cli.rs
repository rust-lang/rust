// revisions: one two three four five
// compile-flags: --crate-type=rlib --target=aarch64-unknown-linux-gnu
// needs-llvm-components: aarch64
//
//
// [one] check-fail
// [one] compile-flags: -C target-feature=+paca
// [two] check-fail
// [two] compile-flags: -C target-feature=-pacg,+pacg
// [three] check-fail
// [three] compile-flags: -C target-feature=+paca,+pacg,-paca
// [four] build-pass
// [four] compile-flags: -C target-feature=-paca,+pacg -C target-feature=+paca
// [five] build-pass
// [five] compile-flags: -C target-feature=+neon
#![feature(no_core, lang_items)]
#![no_core]

#[lang="sized"]
trait Sized {}

fn main() {}
