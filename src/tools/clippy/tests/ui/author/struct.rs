//@ check-pass

#![allow(
    clippy::unnecessary_operation,
    clippy::single_match,
    clippy::no_effect,
    clippy::bool_to_int_with_if
)]
fn main() {
    struct Test {
        field: u32,
    }

    #[clippy::author]
    Test {
        field: if true { 1 } else { 0 },
    };

    let test = Test { field: 1 };

    match test {
        #[clippy::author]
        Test { field: 1 } => {},
        _ => {},
    }

    struct TestTuple(u32);

    let test_tuple = TestTuple(1);

    match test_tuple {
        #[clippy::author]
        TestTuple(1) => {},
        _ => {},
    }

    struct TestMethodCall(u32);

    impl TestMethodCall {
        fn test(&self) {}
    }

    let test_method_call = TestMethodCall(1);

    #[clippy::author]
    test_method_call.test();
}
