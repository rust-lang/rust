#![crate_name = "externcallback"]
#![crate_type = "lib"]

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    pub fn rust_dbg_call(
        cb: extern "C" fn(u64) -> u64,
        data: u64,
    ) -> u64;
}

pub fn fact(n: u64) -> u64 {
    unsafe {
        println!("n = {}", n);
        rust_dbg_call(cb, n)
    }
}

pub extern "C" fn cb(data: u64) -> u64 {
    if data == 1 { data } else { fact(data - 1) * data }
}
