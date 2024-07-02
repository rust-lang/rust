//@ check-pass

// see https://github.com/rust-lang/rust/issues/125833
// notionally, taking the address of a static mut is a safe operation,
// as we only point at it instead of generating a true reference to it
static mut FLAG: bool = false;
fn main() {
    let p = std::ptr::addr_of!(FLAG);
    println!("{p:p}")
}
