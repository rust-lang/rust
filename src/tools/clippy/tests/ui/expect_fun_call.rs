// run-rustfix

#![warn(clippy::expect_fun_call)]
#![allow(clippy::to_string_in_format_args)]

/// Checks implementation of the `EXPECT_FUN_CALL` lint

macro_rules! one {
    () => {
        1
    };
}

fn main() {
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

    let with_none_and_format_with_macro: Option<i32> = None;
    with_none_and_format_with_macro.expect(format!("Error {}: fake error", one!()).as_str());

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

    //Issue #2937
    Some("foo").expect(format!("{} {}", 1, 2).as_ref());

    //Issue #2979 - this should not lint
    {
        let msg = "bar";
        Some("foo").expect(msg);
    }

    {
        fn get_string() -> String {
            "foo".to_string()
        }

        fn get_static_str() -> &'static str {
            "foo"
        }

        fn get_non_static_str(_: &u32) -> &str {
            "foo"
        }

        Some("foo").expect(&get_string());
        Some("foo").expect(get_string().as_ref());
        Some("foo").expect(get_string().as_str());

        Some("foo").expect(get_static_str());
        Some("foo").expect(get_non_static_str(&0));
    }

    //Issue #3839
    Some(true).expect(&format!("key {}, {}", 1, 2));

    //Issue #4912 - the receiver is a &Option
    {
        let opt = Some(1);
        let opt_ref = &opt;
        opt_ref.expect(&format!("{:?}", opt_ref));
    }
}
