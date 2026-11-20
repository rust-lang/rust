//@ check-fail
// Regression test for https://github.com/rust-lang/rust/issues/145779
#![warn(unused_attributes)]
#![feature(register_tool)]
#![feature(sanitize)]

fn main() {
    #[export_name = "x"]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[unsafe(naked)]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[track_caller]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[used]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[target_feature(enable = "x")]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[deprecated]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[inline]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[link_name = "x"]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[link_section = "__TEXT,__text"]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[link_ordinal(42)]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[non_exhaustive]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[proc_macro]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[cold]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[no_mangle]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[deprecated]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[automatically_derived]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[macro_use]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[must_use]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[no_implicit_prelude]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[path = ""]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[ignore]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[should_panic]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[link_name = "x"]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[sanitize(address = "off")]
    //~^ ERROR attribute cannot be used on macro calls
    unreachable!();

    #[repr()]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    //~| WARN unused attribute
    unreachable!();
    #[repr(u8)]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[repr(align(8))]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[repr(packed)]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[repr(C)]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[repr(Rust)]
    //~^ WARN attribute cannot be used on macro calls
    //~| WARN previously accepted
    unreachable!();
    #[repr(simd)]
    //~^ ERROR attribute cannot be used on macro calls
    //~| ERROR SIMD types are experimental and possibly buggy
    unreachable!();
    #[register_tool(xyz)]
    //~^ ERROR crate-level attribute should be an inner attribute
    unreachable!();
    #[deprecated = concat!("woah", "dude")]
    //~^ ERROR attribute value must be a literal
    #[doc = concat!("woah", "dude")]
    //~^ ERROR attribute value must be a literal
    unreachable!();
    #[doc = {
        let a = 1;
        let b = 1;
        let sum = a + b;
        assert_eq!(sum, 2);
    }]
    //~^^^^^^ ERROR attribute value must be a literal
    unreachable!();
    #[doc = {
        let expressions @ r#in @ doc @ attributes @ at = home;
    }]
    //~^^^ ERROR attribute value must be a literal
    unreachable!();
}
