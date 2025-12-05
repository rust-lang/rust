// Validation makes this fail in the wrong place
//@compile-flags: -Zmiri-disable-validation

fn main() {
    let g = unsafe { std::mem::transmute::<usize, fn(i32)>(42) };

    g(42) //~ ERROR: is a dangling pointer
}
