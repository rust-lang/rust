// Validation changes why we fail
// compile-flags: -Zmiri-disable-validation

// error-pattern: tried to deallocate `Stack` memory but gave `Machine(Rust)` as the kind

fn main() {
    let x = 42;
    let bad_box = unsafe { std::mem::transmute::<&i32, Box<i32>>(&x) };
    drop(bad_box);
}
