//@ build-pass
//@ revisions: unsigned_ signed_ default_
//@ [unsigned_] compile-flags: -Zc-char-type=unsigned
//@ [signed_] compile-flags: -Zc-char-type=signed
//@ [default_] compile-flags: -Zc-char-type=default

#![feature(c_char_type)]

#![no_core]
#![crate_type = "rlib"]
#![feature(intrinsics, rustc_attrs, no_core, lang_items, staged_api)]
#![stable(feature = "test", since = "1.0.0")]

#[lang="sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}
impl Copy for bool {}

#[stable(feature = "test", since = "1.0.0")]
#[rustc_const_stable(feature = "test", since = "1.0.0")]
#[rustc_intrinsic]
const unsafe fn unreachable() -> !;

#[rustc_builtin_macro]
macro_rules! cfg {
    ($($cfg:tt)*) => {};
}

const fn do_or_die(cond: bool) {
    if cond {
    } else {
        unsafe { unreachable() }
    }
}

macro_rules! assert {
    ($x:expr $(,)?) => {
        const _: () = do_or_die($x);
    };
}

fn main() {
    #[cfg(unsigned_)]
    assert!(cfg!(c_char_type = "unsigned"));
    #[cfg(signed_)]
    assert!(cfg!(c_char_type = "signed"));
    #[cfg(default_)]
    assert!(cfg!(c_char_type = "default"));
}
