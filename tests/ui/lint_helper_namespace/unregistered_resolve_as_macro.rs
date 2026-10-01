//@check-pass

#![feature(decl_macro)]
#![feature(macro_attr)]

#![crate_type = "lib"]

#[lint::as_ptr]
struct Foo;

mod lint {
    // On stable rust, this could be a re-exported proc macro
    pub macro as_ptr {
        attr() { $($tt:tt)* } => { $($tt)* }
    }
}
