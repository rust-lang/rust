pub fn public_fn(x: u32) -> u32 {
    private_fn(x)
}

#[inline]
fn private_fn(x: u32) -> u32 {
    x + 1
}

pub fn inlined_panic(should_panic: bool) {
    if should_panic {
        panic!("panic at lib.rs:12");
    }
}
