fn main() {
    fn f(_: (i32, i32)) {}

    let g = unsafe { std::mem::transmute::<fn((i32, i32)), fn()>(f) };

    g() //~ ERROR: calling a function with fewer arguments than it requires
}
