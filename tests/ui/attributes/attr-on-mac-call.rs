//@ check-fail
// Regression test for https://github.com/rust-lang/rust/issues/145779
#![warn(unused_attributes)]
#![feature(register_tool)]
#![feature(sanitize)]

fn main() {
    #[export_name = "x"]
    //~^ ERROR attribute cannot be used on macro calls

    #[unsafe(naked)]
    //~^ ERROR attribute cannot be used on macro calls

    #[track_caller]
    //~^ ERROR attribute cannot be used on macro calls

    #[used]
    //~^ ERROR attribute cannot be used on macro calls

    #[target_feature(enable = "x")]
    //~^ ERROR attribute cannot be used on macro calls

    #[deprecated]
    //~^ ERROR attribute cannot be used on macro calls

    #[inline]
    //~^ ERROR attribute cannot be used on macro calls

    #[link_name = "x"]
    //~^ ERROR attribute cannot be used on macro calls

    #[link_section = "__TEXT,__text"]
    //~^ ERROR attribute cannot be used on macro calls

    #[link_ordinal(42)]
    //~^ ERROR attribute cannot be used on macro calls

    #[non_exhaustive]
    //~^ ERROR attribute cannot be used on macro calls

    #[proc_macro]
    //~^ ERROR attribute cannot be used on macro calls

    #[cold]
    //~^ ERROR attribute cannot be used on macro calls

    #[no_mangle]
    //~^ ERROR attribute cannot be used on macro calls

    #[deprecated]
    //~^ ERROR attribute cannot be used on macro calls

    #[automatically_derived]
    //~^ ERROR attribute cannot be used on macro calls

    #[macro_use]
    //~^ ERROR attribute cannot be used on macro calls

    #[must_use]
    //~^ ERROR attribute cannot be used on macro calls

    #[no_implicit_prelude]
    //~^ ERROR attribute cannot be used on macro calls

    #[path = ""]
    //~^ ERROR attribute cannot be used on macro calls

    #[ignore]
    //~^ ERROR attribute cannot be used on macro calls

    #[should_panic]
    //~^ ERROR attribute cannot be used on macro calls

    #[link_name = "x"]
    //~^ ERROR attribute cannot be used on macro calls

    #[sanitize(address = "off")]
    //~^ ERROR attribute cannot be used on macro calls
    unreachable!();

    #[repr()]
    //~^ ERROR attribute cannot be used on macro calls

    //~| WARN unused attribute
    unreachable!();
    #[repr(u8)]
    //~^ ERROR attribute cannot be used on macro calls

    unreachable!();
    #[repr(align(8))]
    //~^ ERROR attribute cannot be used on macro calls

    unreachable!();
    #[repr(packed)]
    //~^ ERROR attribute cannot be used on macro calls

    unreachable!();
    #[repr(C)]
    //~^ ERROR attribute cannot be used on macro calls

    unreachable!();
    #[repr(Rust)]
    //~^ ERROR attribute cannot be used on macro calls

    unreachable!();
    #[repr(simd)]
    //~^ ERROR attribute cannot be used on macro calls

    unreachable!();
    #[register_tool(xyz)]
    //~^ ERROR `#[register_tool]` attribute cannot be used on macro calls
    unreachable!();
}
