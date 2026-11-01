#![warn(clippy::owned_cow)]

use std::borrow::Cow;
use std::ffi::{CString, OsString};
use std::path::PathBuf;

fn main() {
    let x: Cow<'static, String> = Cow::Owned(String::from("Hi!"));
    //~^ ERROR: needlessly owned Cow type
    let y: Cow<'_, Vec<u8>> = Cow::Owned(vec![]);
    //~^ ERROR: needlessly owned Cow type
    let z: Cow<'_, Vec<_>> = Cow::Owned(vec![2_i32]);
    //~^ ERROR: needlessly owned Cow type
    let o: Cow<'_, OsString> = Cow::Owned(OsString::new());
    //~^ ERROR: needlessly owned Cow type
    let c: Cow<'_, CString> = Cow::Owned(CString::new("").unwrap());
    //~^ ERROR: needlessly owned Cow type
    let p: Cow<'_, PathBuf> = Cow::Owned(PathBuf::new());
    //~^ ERROR: needlessly owned Cow type

    // false positive: borrowed type
    let b: Cow<'_, str> = Cow::Borrowed("Hi!");
}
