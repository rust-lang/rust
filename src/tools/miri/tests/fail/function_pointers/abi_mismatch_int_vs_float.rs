fn main() {
    fn f(_: f32) {}

    let g = unsafe { std::mem::transmute::<fn(f32), fn(i32)>(f) };

    g(42) //~ ERROR: calling a function with argument of type f32 passing data of type i32
}
