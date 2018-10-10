fn f() {}

fn main() {
    let x: u8 = unsafe {
        *std::mem::transmute::<fn(), *const u8>(f) //~ ERROR constant evaluation error: tried to dereference a function pointer
    };
    panic!("this should never print: {}", x);
}
