fn f() {}

fn main() {
    let x: u8 = unsafe {
        *std::mem::transmute::<fn(), *const u8>(f) //~ ERROR: out-of-bounds
    };
    panic!("this should never print: {}", x);
}
