#[allow(deprecated, invalid_value)]
fn main() {
    let _ = unsafe { std::mem::zeroed::<fn()>() };
    //~^ ERROR: attempted to zero-initialize type `fn()`, which is invalid
}
