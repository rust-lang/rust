// Validation/SB changes why we fail
// compile-flags: -Zmiri-disable-validation -Zmiri-disable-stacked-borrows

// error-pattern: deallocating `Stack` memory using `Machine(Rust)` deallocation operation

fn main() {
    let x = 42;
    let bad_box = unsafe { std::mem::transmute::<&i32, Box<i32>>(&x) };
    drop(bad_box);
}
