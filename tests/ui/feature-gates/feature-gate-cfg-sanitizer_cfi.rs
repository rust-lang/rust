#[cfg(sanitizer_cfi_generalize_pointers)]
//~^ `cfg(sanitizer_cfi_generalize_pointers)` is experimental
fn foo() {}

#[cfg(sanitizer_cfi_normalize_integers)]
//~^ `cfg(sanitizer_cfi_normalize_integers)` is experimental
fn bar() {}

fn main() {}
