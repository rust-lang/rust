#![feature(llvm_asm)]

fn not_in_miri() -> i32 {
    // Inline assembly definitely does not work in Miri.
    let dummy = 42;
    unsafe {
        llvm_asm!("" : : "r"(&dummy));
    }
    return dummy;
}

fn main() {
    not_in_miri();
    println!("cargo:rerun-if-changed=build.rs");
}
