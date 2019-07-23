// Validation makes this fail in the wrong place
// compile-flags: -Zmiri-disable-validation

fn main() {
    let b = Box::new(42);
    let g = unsafe {
        std::mem::transmute::<&Box<usize>, &fn(i32)>(&b)
    };

    (*g)(42) //~ ERROR tried to treat a memory pointer as a function pointer
}
