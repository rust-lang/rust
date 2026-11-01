// Regression test for <https://github.com/rust-lang/rust/issues/13352>,
// check that the never type can be used as a function argument.
//
//@run-pass

fn foo(_: Box<dyn FnMut()>) {}

fn main() {
    #[expect(unreachable_code)]
    foo(loop {
        std::process::exit(0);
    });
}
