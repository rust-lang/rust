// revisions: rpass1
//
// The corresponding ui test can be found in
// `tests/ui/lint/rfc-2383-lint-reason/expect_unused_inside_impl_block.rs`

#![feature(lint_reasons)]
#![warn(unused)]

struct OneUnused;
struct TwoUnused;

impl OneUnused {
    #[expect(unused)]
    fn unused() {}
}

impl TwoUnused {
    #[expect(unused)]
    fn unused1(){}

    // This unused method has `#[expect(unused)]`, so the compiler should not emit a warning.
    // This ui test was added after a regression in the compiler where it did not recognize multiple
    // `#[expect(unused)]` annotations inside of impl blocks.
    // issue 114416
    #[expect(unused)]
    fn unused2(){}
}

fn main() {
    let _ = OneUnused;
    let _ = TwoUnused;
}
