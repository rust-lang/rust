//@ reference: input.encoding.utf8
//@ reference: input.encoding.invalid

fn foo() {
    include!("not-utf8.bin");
    //~^ ERROR couldn't read `$DIR/not-utf8.bin`: stream did not contain valid UTF-8
}
