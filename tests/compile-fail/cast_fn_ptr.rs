fn main() { //~ ERROR tried to call a function of type
    fn f() {}

    let g = unsafe {
        std::mem::transmute::<fn(), fn(i32)>(f)
    };

    g(42)
}
