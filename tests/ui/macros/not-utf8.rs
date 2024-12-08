//@ error-pattern: did not contain valid UTF-8
//@ reference: input.encoding.utf8
//@ reference: input.encoding.invalid

fn foo() {
    include!("not-utf8.bin")
}
