// run-rustfix

#![allow(clippy::needless_borrow, clippy::ptr_arg)]
#![warn(clippy::unnecessary_to_owned)]
#![feature(custom_inner_attributes)]

use std::borrow::Cow;
use std::ffi::{CStr, CString, OsStr, OsString};
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

#[allow(dead_code)]
#[derive(Clone)]
enum FileType {
    Account,
    PrivateKey,
    Certificate,
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
    let x_ref = &x;

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
    require_str(&x_ref.to_string());

    require_slice(&slice.to_vec());
    require_slice(&Cow::from(slice).into_owned());
    require_slice(&array.to_owned());
    require_slice(&array_ref.to_owned());
    require_slice(&slice.to_owned());
    require_slice(&x_ref.to_owned()); // No longer flagged because of #8759.

    require_x(&Cow::<X>::Owned(x.clone()).into_owned());
    require_x(&x_ref.to_owned()); // No longer flagged because of #8759.

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

    let _ = x.join(&x_ref.to_string());

    let _ = slice.to_vec().into_iter();
    let _ = slice.to_owned().into_iter();
    let _ = [std::path::PathBuf::new()][..].to_vec().into_iter();
    let _ = [std::path::PathBuf::new()][..].to_owned().into_iter();

    let _ = IntoIterator::into_iter(slice.to_vec());
    let _ = IntoIterator::into_iter(slice.to_owned());
    let _ = IntoIterator::into_iter([std::path::PathBuf::new()][..].to_vec());
    let _ = IntoIterator::into_iter([std::path::PathBuf::new()][..].to_owned());

    let _ = check_files(&[FileType::Account]);

    // negative tests
    require_string(&s.to_string());
    require_string(&Cow::from(s).into_owned());
    require_string(&s.to_owned());
    require_string(&x_ref.to_string());

    // `X` isn't copy.
    require_slice(&x.to_owned());
    require_deref_slice(x.to_owned());

    // The following should be flagged by `redundant_clone`, but not by this lint.
    require_c_str(&CString::from_vec_with_nul(vec![0]).unwrap().to_owned());
    require_os_str(&OsString::from("x").to_os_string());
    require_path(&std::path::PathBuf::from("x").to_path_buf());
    require_str(&String::from("x").to_string());
    require_slice(&[String::from("x")].to_owned());
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

// `check_files` is based on:
// https://github.com/breard-r/acmed/blob/1f0dcc32aadbc5e52de6d23b9703554c0f925113/acmed/src/storage.rs#L262
fn check_files(file_types: &[FileType]) -> bool {
    for t in file_types.to_vec() {
        let path = match get_file_path(&t) {
            Ok(p) => p,
            Err(_) => {
                return false;
            },
        };
        if !path.is_file() {
            return false;
        }
    }
    true
}

fn get_file_path(_file_type: &FileType) -> Result<std::path::PathBuf, std::io::Error> {
    Ok(std::path::PathBuf::new())
}

fn require_string(_: &String) {}

fn _msrv_1_35() {
    #![clippy::msrv = "1.35"]
    // `copied` was stabilized in 1.36, so clippy should use `cloned`.
    let _ = &["x"][..].to_vec().into_iter();
}

fn _msrv_1_36() {
    #![clippy::msrv = "1.36"]
    let _ = &["x"][..].to_vec().into_iter();
}

// https://github.com/rust-lang/rust-clippy/issues/8507
mod issue_8507 {
    #![allow(dead_code)]

    struct Opaque<P>(P);

    pub trait Abstracted {}

    impl<P> Abstracted for Opaque<P> {}

    fn build<P>(p: P) -> Opaque<P>
    where
        P: AsRef<str>,
    {
        Opaque(p)
    }

    // Should not lint.
    fn test_str(s: &str) -> Box<dyn Abstracted> {
        Box::new(build(s.to_string()))
    }

    // Should not lint.
    fn test_x(x: super::X) -> Box<dyn Abstracted> {
        Box::new(build(x))
    }

    #[derive(Clone, Copy)]
    struct Y(&'static str);

    impl AsRef<str> for Y {
        fn as_ref(&self) -> &str {
            self.0
        }
    }

    impl ToString for Y {
        fn to_string(&self) -> String {
            self.0.to_string()
        }
    }

    // Should lint because Y is copy.
    fn test_y(y: Y) -> Box<dyn Abstracted> {
        Box::new(build(y.to_string()))
    }
}

// https://github.com/rust-lang/rust-clippy/issues/8759
mod issue_8759 {
    #![allow(dead_code)]

    #[derive(Default)]
    struct View {}

    impl std::borrow::ToOwned for View {
        type Owned = View;
        fn to_owned(&self) -> Self::Owned {
            View {}
        }
    }

    #[derive(Default)]
    struct RenderWindow {
        default_view: View,
    }

    impl RenderWindow {
        fn default_view(&self) -> &View {
            &self.default_view
        }
        fn set_view(&mut self, _view: &View) {}
    }

    fn main() {
        let mut rw = RenderWindow::default();
        rw.set_view(&rw.default_view().to_owned());
    }
}

mod issue_8759_variant {
    #![allow(dead_code)]

    #[derive(Clone, Default)]
    struct View {}

    #[derive(Default)]
    struct RenderWindow {
        default_view: View,
    }

    impl RenderWindow {
        fn default_view(&self) -> &View {
            &self.default_view
        }
        fn set_view(&mut self, _view: &View) {}
    }

    fn main() {
        let mut rw = RenderWindow::default();
        rw.set_view(&rw.default_view().to_owned());
    }
}

mod issue_9317 {
    #![allow(dead_code)]

    struct Bytes {}

    impl ToString for Bytes {
        fn to_string(&self) -> String {
            "123".to_string()
        }
    }

    impl AsRef<[u8]> for Bytes {
        fn as_ref(&self) -> &[u8] {
            &[1, 2, 3]
        }
    }

    fn consume<C: AsRef<[u8]>>(c: C) {
        let _ = c;
    }

    pub fn main() {
        let b = Bytes {};
        // Should not lint.
        consume(b.to_string());
    }
}

mod issue_9351 {
    #![allow(dead_code)]

    use std::ops::Deref;
    use std::path::{Path, PathBuf};

    fn require_deref_path<T: Deref<Target = std::path::Path>>(x: T) -> T {
        x
    }

    fn generic_arg_used_elsewhere<T: AsRef<Path>>(_x: T, _y: T) {}

    fn id<T: AsRef<str>>(x: T) -> T {
        x
    }

    fn predicates_are_satisfied(_x: impl std::fmt::Write) {}

    // Should lint
    fn single_return() -> impl AsRef<str> {
        id("abc".to_string())
    }

    // Should not lint
    fn multiple_returns(b: bool) -> impl AsRef<str> {
        if b {
            return String::new();
        }

        id("abc".to_string())
    }

    struct S1(String);

    // Should not lint
    fn fields1() -> S1 {
        S1(id("abc".to_string()))
    }

    struct S2 {
        s: String,
    }

    // Should not lint
    fn fields2() {
        let mut s = S2 { s: "abc".into() };
        s.s = id("abc".to_string());
    }

    pub fn main() {
        let path = std::path::Path::new("x");
        let path_buf = path.to_owned();

        // Should not lint.
        let _x: PathBuf = require_deref_path(path.to_owned());
        generic_arg_used_elsewhere(path.to_owned(), path_buf);
        predicates_are_satisfied(id("abc".to_string()));
    }
}

mod issue_9504 {
    #![allow(dead_code)]

    async fn foo<S: AsRef<str>>(_: S) {}
    async fn bar() {
        foo(std::path::PathBuf::new().to_string_lossy().to_string()).await;
    }
}

mod issue_9771a {
    #![allow(dead_code)]

    use std::marker::PhantomData;

    pub struct Key<K: AsRef<[u8]>, V: ?Sized>(K, PhantomData<V>);

    impl<K: AsRef<[u8]>, V: ?Sized> Key<K, V> {
        pub fn new(key: K) -> Key<K, V> {
            Key(key, PhantomData)
        }
    }

    pub fn pkh(pkh: &[u8]) -> Key<Vec<u8>, String> {
        Key::new([b"pkh-", pkh].concat().to_vec())
    }
}

mod issue_9771b {
    #![allow(dead_code)]

    pub struct Key<K: AsRef<[u8]>>(K);

    pub fn from(c: &[u8]) -> Key<Vec<u8>> {
        let v = [c].concat();
        Key(v.to_vec())
    }
}
