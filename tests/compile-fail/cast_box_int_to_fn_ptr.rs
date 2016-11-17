fn main() {
    let b = Box::new(42);
    let g = unsafe {
        std::mem::transmute::<&usize, &fn(i32)>(&b)
    };

    (*g)(42) //~ ERROR tried to use an integer pointer or a dangling pointer as a function pointer
}
