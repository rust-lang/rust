// run-pass
// min-llvm-version: 13.0
// compiler-flags: -O

// Regression test for issue #80309

pub unsafe fn foo(x: *const i8) -> i8 {
    *x.wrapping_sub(x as _).wrapping_add(x as _)
}

fn main() {
    let x = 42;
    println!("{}", unsafe { foo(&x) });
}
