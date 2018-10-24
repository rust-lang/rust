// Validation makes this fail in the wrong place
// compile-flags: -Zmiri-disable-validation

fn main() {
    let b = Box::new(42);
    let g = unsafe {
        std::mem::transmute::<&usize, &fn(i32)>(&b)
    };

    (*g)(42) //~ ERROR a memory access tried to interpret some bytes as a pointer
}
