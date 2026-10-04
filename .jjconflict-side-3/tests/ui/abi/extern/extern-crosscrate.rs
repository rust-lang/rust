//@ run-pass
//@ aux-build:extern-crosscrate-source.rs

extern crate externcallback;

fn fact(n: u64) -> u64 {
    unsafe {
        println!("n = {:?}", n);
        externcallback::rust_dbg_call(externcallback::cb, n)
    }
}

pub fn main() {
    let result = fact(10);
    println!("result = {}", result);
    assert_eq!(result, 3628800);
}
