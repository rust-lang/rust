//@ known-bug: rust-lang/rust#129109
//@ compile-flags: -Zmir-opt-level=5 -Zvalidate-mir

extern "C" {
    pub static mut symbol: [i8];
}

fn main() {
    println!("C", unsafe { &symbol });
}
