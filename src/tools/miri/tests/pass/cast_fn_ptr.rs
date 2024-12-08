fn main() {
    fn f(_: *const u8) {}

    let g = unsafe { std::mem::transmute::<fn(*const u8), fn(*const i32)>(f) };

    g(&42 as *const _);
}
