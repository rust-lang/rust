// rust-lang/rust#61232: We used to ICE when trying to detect a
// collision on the symbol generated for the external linkage item in
// an extern crate.

//@ build-fail
//@ aux-build:def_colliding_external.rs
// FIXME(#83838) codegen-units=1 triggers llvm asserts
//@ compile-flags: -Ccodegen-units=16

extern crate def_colliding_external as dep1;

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

//~? ERROR symbol `collision` is already defined
