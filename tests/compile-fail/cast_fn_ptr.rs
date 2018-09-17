fn main() {
    fn f() {}

    let g = unsafe {
        std::mem::transmute::<fn(), fn(i32)>(f)
    };

    g(42) //~ ERROR tried to call a function with incorrect number of arguments
}
