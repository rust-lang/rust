fn main() {
    let x : i16 = 22;
    ((&x) as *const i16) as f32;
    //~^ ERROR casting `*const i16` as `f32` is invalid
}
