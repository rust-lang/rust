// check-pass
// compile-flags: --crate-type=lib

#![feature(decl_macro)]
#![feature(lang_items)]
#![feature(no_core)]
#![feature(rustc_attrs)]
#![no_core]

#[rustc_builtin_macro]
macro derive() {}

#[rustc_builtin_macro(Default, attributes(default))]
macro Default() {}

mod default {
    pub trait Default {
        fn default() -> Self;
    }

    impl Default for u8 {
        fn default() -> u8 {
            0
        }
    }
}

#[lang = "drop_in_place"]
#[allow(unconditional_recursion)]
pub unsafe fn drop_in_place<T: ?Sized>(to_drop: *mut T) {
    drop_in_place(to_drop);
}

#[lang = "copy"]
trait Copy {}

#[lang = "sized"]
trait Sized {}

#[derive(Default)]
enum S {
    #[default] // OK
    Foo,
}
