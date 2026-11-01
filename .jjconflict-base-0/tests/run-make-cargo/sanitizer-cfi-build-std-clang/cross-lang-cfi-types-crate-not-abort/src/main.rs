use cfi_types::c_long;

#[link(name = "foo")]
extern "C" {
    // This declaration has the type id "_ZTSFvlE" because it uses the CFI types
    // for cross-language LLVM CFI support. The cfi_types crate provides a new
    // set of C types as user-defined types using the cfi_encoding attribute and
    // repr(transparent) to be used for cross-language LLVM CFI support. This
    // new set of C types allows the Rust compiler to identify and correctly
    // encode C types in extern "C" function types indirectly called across the
    // FFI boundary when CFI is enabled.
    fn hello_from_c(_: c_long);

    // This declaration has the type id "_ZTSFvPFvlElE" because it uses the CFI
    // types for cross-language LLVM CFI support--this can be ignored for the
    // purposes of this example.
    fn indirect_call_from_c(f: unsafe extern "C" fn(c_long), arg: c_long);
}

// This definition has the type id "_ZTSFvlE" because it uses the CFI types for
// cross-language LLVM CFI support, similarly to the hello_from_c declaration
// above.
unsafe extern "C" fn hello_from_rust(_: c_long) {
    println!("Hello, world!");
}

// This definition has the type id "_ZTSFvlE" because it uses the CFI types for
// cross-language LLVM CFI support, similarly to the hello_from_c declaration
// above.
unsafe extern "C" fn hello_from_rust_again(_: c_long) {
    println!("Hello from Rust again!");
}

// This definition would also have the type id "_ZTSFvPFvlElE" because it uses
// the CFI types for cross-language LLVM CFI support, similarly to the
// hello_from_c declaration above--this can be ignored for the purposes of this
// example.
fn indirect_call(f: unsafe extern "C" fn(c_long), arg: c_long) {
    // This indirect call site tests whether the destination pointer is a member
    // of the group derived from the same type id of the f declaration, which
    // has the type id "_ZTSFvlE" because it uses the CFI types for
    // cross-language LLVM CFI support, similarly to the hello_from_c
    // declaration above.
    unsafe { f(arg) }
}

// This definition has the type id "_ZTSFvvE"--this can be ignored for the
// purposes of this example.
fn main() {
    // This demonstrates an indirect call within Rust-only code using the same
    // encoding for hello_from_rust and the test at the indirect call site at
    // indirect_call (i.e., "_ZTSFvlE").
    indirect_call(hello_from_rust, c_long(5));

    // This demonstrates an indirect call across the FFI boundary with the Rust
    // compiler and Clang using the same encoding for hello_from_c and the test
    // at the indirect call site at indirect_call (i.e., "_ZTSFvlE").
    indirect_call(hello_from_c, c_long(5));

    // This demonstrates an indirect call to a function passed as a callback
    // across the FFI boundary with the Rust compiler and Clang the same
    // encoding for the passed-callback declaration and the test at the indirect
    // call site at indirect_call_from_c (i.e., "_ZTSFvlE").
    unsafe {
        indirect_call_from_c(hello_from_rust_again, c_long(5));
    }
}
