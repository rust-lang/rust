//@ run-pass
//@ ignore-emscripten blows the JS stack

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    pub fn rust_dbg_call(
        cb: extern "C" fn(u64) -> u64,
        data: u64,
    ) -> u64;
}

extern "C" fn cb(data: u64) -> u64 {
    if data == 1 { data } else { count(data - 1) + 1 }
}

fn count(n: u64) -> u64 {
    unsafe {
        println!("n = {}", n);
        rust_dbg_call(cb, n)
    }
}

pub fn main() {
    let result = count(1000);
    println!("result = {:?}", result);
    assert_eq!(result, 1000);
}
