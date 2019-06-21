// Validation makes this fail in the wrong place
// compile-flags: -Zmiri-disable-validation -Zmiri-seed=0000000000000000

fn main() {
    let g = unsafe {
        std::mem::transmute::<usize, fn(i32)>(42)
    };

    g(42) //~ ERROR dangling pointer was dereferenced
}
