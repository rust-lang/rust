// run-fail
// ignore-wasm32-bare: No panic messages
// compile-flags: -C debug-assertions -Zmir-opt-level=0

#[repr(C)]
struct Thing {
    x: usize,
    y: Contents,
    z: usize,
}

#[repr(usize)]
enum Contents {
    A = 8usize,
    B = 9usize,
    C = 10usize,
}

fn main() {
    unsafe {
        let _thing = std::mem::transmute::<(usize, usize, usize), Thing>((0, 3, 0));
    }
}
