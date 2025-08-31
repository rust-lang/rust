//! Regression test for: https://github.com/rust-lang/rust/issues/144792

fn test_cow_suggestion() -> String {
    let os_string = std::ffi::OsString::from("test");
    os_string.to_string_lossy().to_owned()
    //~^ ERROR use of `.to_owned()` on `Cow`
    //~| HELP try using `.into_owned()`
}

// Test multiple Cow scenarios
fn test_cow_from_str() -> String {
    let s = "hello";
    let cow = std::borrow::Cow::from(s);
    cow.to_owned() // Should suggest into_owned()
    //~^ ERROR use of `.to_owned()` on `Cow`
    //~| HELP try using `.into_owned()`
}

// Test with different Cow types
fn test_cow_bytes() -> Vec<u8> {
    let bytes = b"hello";
    let cow = std::borrow::Cow::from(&bytes[..]);
    cow.to_owned() // Should suggest into_owned()
    //~^ ERROR use of `.to_owned()` on `Cow`
    //~| HELP try using `.into_owned()`
}
