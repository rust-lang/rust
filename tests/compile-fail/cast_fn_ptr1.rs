fn main() {
    fn f() {} //~ ERROR calling a function with more arguments than it expected

    let g = unsafe {
        std::mem::transmute::<fn(), fn(i32)>(f)
    };

    g(42)
}
