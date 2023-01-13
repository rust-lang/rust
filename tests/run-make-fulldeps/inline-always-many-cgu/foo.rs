#![crate_type = "lib"]

pub mod a {
    #[inline(always)]
    pub fn foo() {
    }

    pub fn bar() {
    }
}

#[no_mangle]
pub fn bar() {
    a::foo();
}
