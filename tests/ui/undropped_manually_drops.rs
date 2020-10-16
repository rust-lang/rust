#![warn(clippy::undropped_manually_drops)]

struct S;

fn main() {
    let f = drop;
    let manual = std::mem::ManuallyDrop::new(S);

    // These lines will not drop `S`
    drop(std::mem::ManuallyDrop::new(S));
    f(manual);

    // These lines will
    unsafe {
        std::mem::ManuallyDrop::drop(std::mem::ManuallyDrop::new(S));
        std::mem::ManuallyDrop::drop(manual);
    }
}
