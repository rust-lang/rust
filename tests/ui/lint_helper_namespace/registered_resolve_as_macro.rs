//@check-pass

#![crate_type = "lib"]

#![feature(decl_macro)]
#![feature(macro_attr)]
#![feature(register_tool)]
#![register_tool(lint)]

#[lint::as_ptr]
struct Foo;

mod lint {
    // On stable rust, this could be a re-exported proc macro
    pub macro as_ptr {
        attr() { $($tt:tt)* } => { $($tt)* }
    }
}
