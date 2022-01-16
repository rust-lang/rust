// check-pass
fn assert_debug<T: std::fmt::Debug>() {}

fn main() {
    assert_debug::<extern "C" fn()>();
    assert_debug::<extern "C" fn(argument: u32)>();
    assert_debug::<unsafe extern "C" fn(argument: u32)>();
    assert_debug::<unsafe extern "C" fn(argument: u32, ...)>();
    assert_debug::<fn(argument: u32)>();
    assert_debug::<extern "fastcall" fn(argument: u32)>();
}
