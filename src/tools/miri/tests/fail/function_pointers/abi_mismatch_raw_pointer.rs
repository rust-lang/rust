fn main() {
    fn f(_: *const [i32]) {}

    let g = unsafe { std::mem::transmute::<fn(*const [i32]), fn(*const i32)>(f) };

    g(&42 as *const i32) //~ ERROR: type *const [i32] passing argument of type *const i32
}
