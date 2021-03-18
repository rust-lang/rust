// Test that the compiler will catch invalid inline assembly constraints.

// build-fail
// ignore-emscripten

#![feature(llvm_asm)]

extern "C" {
    fn foo(a: usize);
}

fn main() {
    bad_register_constraint();
    bad_input();
    wrong_size_output();
}

// Issue #54130
fn bad_register_constraint() {
    let rax: u64;
    unsafe {
        llvm_asm!("" :"={rax"(rax)) //~ ERROR E0668
    };
    println!("Accumulator is: {}", rax);
}

// Issue #54376
fn bad_input() {
    unsafe {
        llvm_asm!("callq $0" : : "0"(foo)) //~ ERROR E0668
    };
}

fn wrong_size_output() {
    let rax: u64 = 0;
    unsafe {
        llvm_asm!("addb $1, $0" : "={rax}"((0i32, rax))); //~ ERROR E0668
    }
    println!("rax: {}", rax);
}
