#[allow(deprecated, invalid_value)]
fn main() {
    let _ = unsafe { std::mem::zeroed::<fn()>() }; //~ERROR: constructing invalid value
}
