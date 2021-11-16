// run-rustfix

#![allow(clippy::ptr_arg)]
// Some of the expressions that `redundant_clone` flags overlap with ours. Enabling it interferes
// with `rustfix`.
#![allow(clippy::redundant_clone)]
// `needless_borrow` is for checking the fixed code.
#![warn(clippy::needless_borrow)]
#![warn(clippy::unnecessary_to_owned)]

use std::borrow::Cow;
use std::ffi::{CStr, OsStr};
use std::ops::Deref;

#[derive(Clone)]
struct X(String);

impl Deref for X {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

impl AsRef<str> for X {
    fn as_ref(&self) -> &str {
        self.0.as_str()
    }
}

impl ToString for X {
    fn to_string(&self) -> String {
        self.0.to_string()
    }
}

impl X {
    fn join(&self, other: impl AsRef<str>) -> Self {
        let mut s = self.0.clone();
        s.push_str(other.as_ref());
        Self(s)
    }
}

fn main() {
    let c_str = CStr::from_bytes_with_nul(&[0]).unwrap();
    let os_str = OsStr::new("x");
    let path = std::path::Path::new("x");
    let s = "x";
    let array = ["x"];
    let array_ref = &["x"];
    let slice = &["x"][..];
    let x = X(String::from("x"));

    require_c_str(&Cow::from(c_str).into_owned());
    require_c_str(&c_str.to_owned());

    require_os_str(&os_str.to_os_string());
    require_os_str(&Cow::from(os_str).into_owned());
    require_os_str(&os_str.to_owned());

    require_path(&path.to_path_buf());
    require_path(&Cow::from(path).into_owned());
    require_path(&path.to_owned());

    require_str(&s.to_string());
    require_str(&Cow::from(s).into_owned());
    require_str(&s.to_owned());
    require_str(&x.to_string());

    require_slice(&slice.to_vec());
    require_slice(&Cow::from(slice).into_owned());
    require_slice(&array.to_owned());
    require_slice(&array_ref.to_owned());
    require_slice(&slice.to_owned());
    require_slice(&x.to_owned());

    require_x(&Cow::<X>::Owned(x.clone()).into_owned());
    require_x(&x.to_owned());

    require_deref_c_str(c_str.to_owned());
    require_deref_os_str(os_str.to_owned());
    require_deref_path(path.to_owned());
    require_deref_str(s.to_owned());
    require_deref_slice(slice.to_owned());

    require_impl_deref_c_str(c_str.to_owned());
    require_impl_deref_os_str(os_str.to_owned());
    require_impl_deref_path(path.to_owned());
    require_impl_deref_str(s.to_owned());
    require_impl_deref_slice(slice.to_owned());

    require_deref_str_slice(s.to_owned(), slice.to_owned());
    require_deref_slice_str(slice.to_owned(), s.to_owned());

    require_as_ref_c_str(c_str.to_owned());
    require_as_ref_os_str(os_str.to_owned());
    require_as_ref_path(path.to_owned());
    require_as_ref_str(s.to_owned());
    require_as_ref_str(x.to_owned());
    require_as_ref_slice(array.to_owned());
    require_as_ref_slice(array_ref.to_owned());
    require_as_ref_slice(slice.to_owned());

    require_impl_as_ref_c_str(c_str.to_owned());
    require_impl_as_ref_os_str(os_str.to_owned());
    require_impl_as_ref_path(path.to_owned());
    require_impl_as_ref_str(s.to_owned());
    require_impl_as_ref_str(x.to_owned());
    require_impl_as_ref_slice(array.to_owned());
    require_impl_as_ref_slice(array_ref.to_owned());
    require_impl_as_ref_slice(slice.to_owned());

    require_as_ref_str_slice(s.to_owned(), array.to_owned());
    require_as_ref_str_slice(s.to_owned(), array_ref.to_owned());
    require_as_ref_str_slice(s.to_owned(), slice.to_owned());
    require_as_ref_slice_str(array.to_owned(), s.to_owned());
    require_as_ref_slice_str(array_ref.to_owned(), s.to_owned());
    require_as_ref_slice_str(slice.to_owned(), s.to_owned());

    let _ = x.join(&x.to_string());

    // negative tests
    require_string(&s.to_string());
    require_string(&Cow::from(s).into_owned());
    require_string(&s.to_owned());
    require_string(&x.to_string());

    // `X` isn't copy.
    require_deref_slice(x.to_owned());
}

fn require_c_str(_: &CStr) {}
fn require_os_str(_: &OsStr) {}
fn require_path(_: &std::path::Path) {}
fn require_str(_: &str) {}
fn require_slice<T>(_: &[T]) {}
fn require_x(_: &X) {}

fn require_deref_c_str<T: Deref<Target = CStr>>(_: T) {}
fn require_deref_os_str<T: Deref<Target = OsStr>>(_: T) {}
fn require_deref_path<T: Deref<Target = std::path::Path>>(_: T) {}
fn require_deref_str<T: Deref<Target = str>>(_: T) {}
fn require_deref_slice<T, U: Deref<Target = [T]>>(_: U) {}

fn require_impl_deref_c_str(_: impl Deref<Target = CStr>) {}
fn require_impl_deref_os_str(_: impl Deref<Target = OsStr>) {}
fn require_impl_deref_path(_: impl Deref<Target = std::path::Path>) {}
fn require_impl_deref_str(_: impl Deref<Target = str>) {}
fn require_impl_deref_slice<T>(_: impl Deref<Target = [T]>) {}

fn require_deref_str_slice<T: Deref<Target = str>, U, V: Deref<Target = [U]>>(_: T, _: V) {}
fn require_deref_slice_str<T, U: Deref<Target = [T]>, V: Deref<Target = str>>(_: U, _: V) {}

fn require_as_ref_c_str<T: AsRef<CStr>>(_: T) {}
fn require_as_ref_os_str<T: AsRef<OsStr>>(_: T) {}
fn require_as_ref_path<T: AsRef<std::path::Path>>(_: T) {}
fn require_as_ref_str<T: AsRef<str>>(_: T) {}
fn require_as_ref_slice<T, U: AsRef<[T]>>(_: U) {}

fn require_impl_as_ref_c_str(_: impl AsRef<CStr>) {}
fn require_impl_as_ref_os_str(_: impl AsRef<OsStr>) {}
fn require_impl_as_ref_path(_: impl AsRef<std::path::Path>) {}
fn require_impl_as_ref_str(_: impl AsRef<str>) {}
fn require_impl_as_ref_slice<T>(_: impl AsRef<[T]>) {}

fn require_as_ref_str_slice<T: AsRef<str>, U, V: AsRef<[U]>>(_: T, _: V) {}
fn require_as_ref_slice_str<T, U: AsRef<[T]>, V: AsRef<str>>(_: U, _: V) {}

fn require_string(_: &String) {}
