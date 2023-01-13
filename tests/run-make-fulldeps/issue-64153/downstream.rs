extern crate upstream;

#[no_mangle]
pub extern "C" fn foo() {
    print!("1 + 1 = {}", upstream::issue64153_test_function(1));
}
