fn main() {
    let g = unsafe {
        std::mem::transmute::<usize, fn(i32)>(42)
    };

    g(42) //~ ERROR tried to use an integer pointer or a dangling pointer as a function pointer
}
