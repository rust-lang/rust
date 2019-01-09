#![warn(clippy::expect_fun_call)]
#![allow(clippy::useless_format)]

/// Checks implementation of the `EXPECT_FUN_CALL` lint

fn expect_fun_call() {
    struct Foo;

    impl Foo {
        fn new() -> Self {
            Foo
        }

        fn expect(&self, msg: &str) {
            panic!("{}", msg)
        }
    }

    let with_some = Some("value");
    with_some.expect("error");

    let with_none: Option<i32> = None;
    with_none.expect("error");

    let error_code = 123_i32;
    let with_none_and_format: Option<i32> = None;
    with_none_and_format.expect(&format!("Error {}: fake error", error_code));

    let with_none_and_as_str: Option<i32> = None;
    with_none_and_as_str.expect(format!("Error {}: fake error", error_code).as_str());

    let with_ok: Result<(), ()> = Ok(());
    with_ok.expect("error");

    let with_err: Result<(), ()> = Err(());
    with_err.expect("error");

    let error_code = 123_i32;
    let with_err_and_format: Result<(), ()> = Err(());
    with_err_and_format.expect(&format!("Error {}: fake error", error_code));

    let with_err_and_as_str: Result<(), ()> = Err(());
    with_err_and_as_str.expect(format!("Error {}: fake error", error_code).as_str());

    let with_dummy_type = Foo::new();
    with_dummy_type.expect("another test string");

    let with_dummy_type_and_format = Foo::new();
    with_dummy_type_and_format.expect(&format!("Error {}: fake error", error_code));

    let with_dummy_type_and_as_str = Foo::new();
    with_dummy_type_and_as_str.expect(format!("Error {}: fake error", error_code).as_str());

    //Issue #2979 - this should not lint
    let msg = "bar";
    Some("foo").expect(msg);

    Some("foo").expect({ &format!("error") });
    Some("foo").expect(format!("error").as_ref());
}

fn main() {}
