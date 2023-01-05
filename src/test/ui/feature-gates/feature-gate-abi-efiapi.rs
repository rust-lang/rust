// needs-llvm-components: x86
// compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=rlib
#![no_core]
#![feature(no_core, lang_items)]
#[lang="sized"]
trait Sized { }

// Functions
extern "efiapi" fn f1() {} //~ ERROR efiapi ABI is experimental

// Methods in trait defintion
trait Tr {
    extern "efiapi" fn f2(); //~ ERROR efiapi ABI is experimental
    extern "efiapi" fn f3() {} //~ ERROR efiapi ABI is experimental
}

struct S;

// Methods in trait impl
impl Tr for S {
    extern "efiapi" fn f2() {} //~ ERROR efiapi ABI is experimental
}

// Methods in inherent impl
impl S {
    extern "efiapi" fn f4() {} //~ ERROR efiapi ABI is experimental
}

// Function pointer types
type A = extern "efiapi" fn(); //~ ERROR efiapi ABI is experimental

// Foreign modules
extern "efiapi" {} //~ ERROR efiapi ABI is experimental
