//! Check that symbol names with pattern types in them are
//! different from the same symbol with the base type

//@ compile-flags: -Csymbol-mangling-version=v0 -Copt-level=0 --crate-type=lib

#![feature(pattern_types)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

type NanoU32 = crate::pattern_type!(u32 is 0..=999_999_999);

fn foo<T>() {}

pub fn bar() {
    // CHECK: call pattern_type_symbols::foo::<u32>
    // CHECK: call void @_RINvC[[CRATE_IDENT:[a-zA-Z0-9]{12}]]_20pattern_type_symbols3foomEB2_
    foo::<u32>();
    // CHECK: call pattern_type_symbols::foo::<u32 is 0..=999999999>
    // CHECK: call void @_RINvC[[CRATE_IDENT]]_20pattern_type_symbols3fooWmRm0_m3b9ac9ff_EB2_
    foo::<NanoU32>();
}
