fn main() {
    fn f() {}

    let g = unsafe { std::mem::transmute::<fn(), fn(i32)>(f) };

    g(42) //~ ERROR: calling a function with more arguments than it expected
}
