//! Regression test for: https://github.com/rust-lang/rust/issues/144792

fn main() {
    let _ = || {
        let os_string = std::ffi::OsString::from("test");
        os_string.to_string_lossy().to_owned()
        //~^ ERROR: cannot return value referencing local variable `os_string` [E0515]
    };

    let _ = || {
        let s = "hello".to_owned();
        let cow = std::borrow::Cow::from(&s);
        cow.to_owned()
        //~^ ERROR: cannot return value referencing local variable `s` [E0515]
    };

    let _ = || {
        let bytes = b"hello".to_owned();
        let cow = std::borrow::Cow::from(&bytes[..]);
        cow.to_owned()
        //~^ ERROR: cannot return value referencing local variable `bytes` [E0515]
    };
}
