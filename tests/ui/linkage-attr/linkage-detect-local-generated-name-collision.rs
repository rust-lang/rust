//@ build-fail
// FIXME(#83838) codegen-units=1 triggers llvm asserts
// FIXME(#144940) `-Zcodegen-source-order` avoids `_rust_extern_with_linkage_collision`
//                sorted after the main, otherwise it causes an ICE. Remove this option
//                once this issue fixed.
//@ compile-flags: -Ccodegen-units=16 -Zcodegen-source-order
#![feature(linkage)]

mod dep1 {
    extern "C" {
        #[linkage = "external"]
        #[no_mangle]
        pub static collision: *const i32; //~ ERROR symbol `collision` is already defined
    }
}

#[no_mangle]
pub static _rust_extern_with_linkage_collision: i32 = 0;

mod dep2 {
    #[no_mangle]
    pub static collision: usize = 0;
}

fn main() {
    unsafe {
        println!("{:p}", &dep1::collision);
    }
}
