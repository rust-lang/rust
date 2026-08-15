fn main() {
    fn f(_: (i32, i32)) {} //~ ERROR: calling a function with fewer arguments than it requires

    let g = unsafe { std::mem::transmute::<fn((i32, i32)), fn()>(f) };

    g()
}
