// revisions: one two three
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
#![feature(no_core, lang_items)]
#![no_core]

#[lang="sized"]
trait Sized {}

fn main() {}
