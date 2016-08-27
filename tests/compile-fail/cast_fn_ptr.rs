fn main() {
    fn f() {}

    let g = unsafe { //~ ERROR tried to call a function of type
        std::mem::transmute::<fn(), fn(i32)>(f)
    };

    g(42)
}
