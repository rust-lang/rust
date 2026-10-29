//@ compile-flags: -Zunpretty=normal
//@ check-pass

#[no_mangle]
extern "C" fn foo() {}

#[unsafe(no_mangle)]
extern "C" fn bar() {}

#[cfg_attr(FALSE, unsafe(no_mangle))]
extern "C" fn zoo() {}
