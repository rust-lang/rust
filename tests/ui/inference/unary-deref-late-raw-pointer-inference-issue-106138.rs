//! Regression test for https://github.com/rust-lang/rust/issues/106138.
//! An unresolved dereference must not treat a raw pointer as an overloaded `Deref`.

fn make<T>() -> T {
    loop {}
}

fn main() {
    let pointer = make();
    //~^ ERROR type annotations needed
    let value = unsafe { *pointer };
    let _: *const u8 = pointer;
    let _: u8 = value;
}
