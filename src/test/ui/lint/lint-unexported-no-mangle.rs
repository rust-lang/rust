// compile-flags:-F private_no_mangle_fns -F no_mangle_const_items -F private_no_mangle_statics

#[no_mangle]
fn foo() {
}

#[allow(dead_code)]
#[no_mangle]
const FOO: u64 = 1; //~ ERROR const items should never be #[no_mangle]

#[no_mangle]
pub const PUB_FOO: u64 = 1; //~ ERROR const items should never be #[no_mangle]

#[no_mangle]
pub fn bar()  {
}

#[no_mangle]
pub static BAR: u64 = 1;

#[allow(dead_code)]
#[no_mangle]
static PRIVATE_BAR: u64 = 1;


fn main() {
    foo();
    bar();
}
