fn main() {
    let x = 42;
    let bad_box = unsafe { std::mem::transmute::<&i32, Box<i32>>(&x) };
    drop(bad_box); //~ ERROR tried to deallocate Stack memory but gave Rust as the kind
}
