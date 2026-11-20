#![allow(clippy::disallowed_names)]

fn foo() -> Option<u32> {
    Some(1)
}

fn helper_function() {
    // Should not lint in integration test file, see https://github.com/rust-lang/rust-clippy/issues/13981
    let baz = foo().unwrap();
    println!("baz: {baz}");
}

#[test]
fn integration_test() {
    helper_function();

    // Should not lint in integration test file, see https://github.com/rust-lang/rust-clippy/issues/13981
    let bar = foo().unwrap();
    println!("bar: {bar}");
}
