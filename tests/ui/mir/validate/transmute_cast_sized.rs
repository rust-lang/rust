//@ build-pass
//@ compile-flags: -Zvalidate-mir
//@ edition: 2021

#![crate_type = "lib"]

// Use `PhantomData` to get target-independent size
async fn get(_r: std::marker::PhantomData<&i32>) {
    loop {}
}

pub fn check() {
    let mut v = get(loop {});
    let _ = || unsafe {
        v = std::mem::transmute([0_u8; 1]);
    };
}
