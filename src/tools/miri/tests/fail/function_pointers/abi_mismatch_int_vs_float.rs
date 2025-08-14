fn main() {
    fn f(_: f32) {}

    let g = unsafe { std::mem::transmute::<fn(f32), fn(i32)>(f) };

    g(42) //~ ERROR: type f32 passing argument of type i32
}
