//@error-in-other-file: did not contain valid UTF-8

fn foo() {
    include!("not-utf8.bin")
}
