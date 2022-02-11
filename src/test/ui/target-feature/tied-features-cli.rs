// only-aarch64
// revisions: one two three four
//[one] compile-flags: -C target-feature=+paca
//[two] compile-flags: -C target-feature=-pacg,+pacg
//[three] compile-flags: -C target-feature=+paca,+pacg,-paca
//[four] check-pass
//[four] compile-flags: -C target-feature=-paca,+pacg -C target-feature=+paca

fn main() {}
