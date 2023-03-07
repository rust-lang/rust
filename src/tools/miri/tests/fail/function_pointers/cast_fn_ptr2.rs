fn main() {
    fn f(_: (i32, i32)) {}

    let g = unsafe { std::mem::transmute::<fn((i32, i32)), fn(i32)>(f) };

    g(42) //~ ERROR: calling a function with argument of type (i32, i32) passing data of type i32
}
