fn main() {
    fn f(_ : (i32,i32)) {}

    let g = unsafe {
        std::mem::transmute::<fn((i32,i32)), fn(i32)>(f)
    };

    g(42) //~ ERROR tried to call a function with sig fn((i32, i32)) through a function pointer of type fn(i32)
}
