//@ check-pass
// Regression test for https://github.com/rust-lang/rust/issues/145779
#![warn(unused_attributes)]

fn main() {
    #[export_name = "x"]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[unsafe(naked)]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[track_caller]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[used]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[target_feature(enable = "x")]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[deprecated]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[inline]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[link_name = "x"]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[link_section = "x"]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[link_ordinal(42)]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[non_exhaustive]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[proc_macro]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[cold]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[no_mangle]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[deprecated]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[automatically_derived]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[macro_use]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[must_use]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[no_implicit_prelude]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[path = ""]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[ignore]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    #[should_panic]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
}
