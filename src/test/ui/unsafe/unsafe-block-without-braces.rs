fn main() {
    unsafe //{
        std::mem::transmute::<f32, u32>(1.0);
    //}
}
//~^^^ ERROR expected one of `extern`, `fn`, or `{`, found `std`
