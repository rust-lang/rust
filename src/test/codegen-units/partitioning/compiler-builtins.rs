// Verifies that during compiler_builtins compilation the codegen units are kept
// unmerged. Even when only a single codegen unit is requested with -Ccodegen-units=1.
//
// compile-flags: -Zprint-mono-items=eager -Ccodegen-units=1

#![compiler_builtins]
#![crate_type="lib"]
#![feature(compiler_builtins)]

mod atomics {
    //~ MONO_ITEM fn compiler_builtins::atomics[0]::sync_1[0] @@ compiler_builtins-cgu.0[External]
    #[no_mangle]
    pub extern "C" fn sync_1() {}

    //~ MONO_ITEM fn compiler_builtins::atomics[0]::sync_2[0] @@ compiler_builtins-cgu.0[External]
    #[no_mangle]
    pub extern "C" fn sync_2() {}

    //~ MONO_ITEM fn compiler_builtins::atomics[0]::sync_3[0] @@ compiler_builtins-cgu.0[External]
    #[no_mangle]
    pub extern "C" fn sync_3() {}
}

mod x {
    //~ MONO_ITEM fn compiler_builtins::x[0]::x[0] @@ compiler_builtins-cgu.1[External]
    #[no_mangle]
    pub extern "C" fn x() {}
}

mod y {
    //~ MONO_ITEM fn compiler_builtins::y[0]::y[0] @@ compiler_builtins-cgu.2[External]
    #[no_mangle]
    pub extern "C" fn y() {}
}

mod z {
    //~ MONO_ITEM fn compiler_builtins::z[0]::z[0] @@ compiler_builtins-cgu.3[External]
    #[no_mangle]
    pub extern "C" fn z() {}
}
