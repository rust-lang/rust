use std::mem::ManuallyDrop;

fn main() {
    let mut x = ManuallyDrop::new(Box::new(1));
    unsafe { ManuallyDrop::drop(&mut x) };
    let _x = x; // move
}
