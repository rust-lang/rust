//@ check-pass

#![deny(unfulfilled_lint_expectations)]
#![warn(dead_code)]

struct Foo {
    #[expect(dead_code)]
    value: usize,
}

#[expect(dead_code)]
fn dead_reads_field() {
    let foo = Foo { value: 0 };
    let _ = foo.value;
}

fn main() {
    let _ = Foo { value: 0 };
}
