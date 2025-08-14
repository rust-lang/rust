fn main() {
    fn f() -> u32 {
        42
    }

    let g = unsafe { std::mem::transmute::<fn() -> u32, fn()>(f) };

    g() //~ ERROR: calling a function with return type u32 passing return place of type ()
}
