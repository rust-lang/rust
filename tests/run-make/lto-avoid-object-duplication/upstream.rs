// Make this function extern "C", public, and no-mangle, so that it gets
// exported from the downstream staticlib.
#[no_mangle]
pub extern "C" fn issue64153_test_function(x: u32) -> u32 {
    x + 1
}
