#![warn(clippy::undropped_manually_drops)]

struct S;

fn main() {
    let f = std::mem::drop;
    let g = std::mem::ManuallyDrop::drop;
    let mut manual1 = std::mem::ManuallyDrop::new(S);
    let mut manual2 = std::mem::ManuallyDrop::new(S);
    let mut manual3 = std::mem::ManuallyDrop::new(S);
    let mut manual4 = std::mem::ManuallyDrop::new(S);

    // These lines will not drop `S` and should be linted
    drop(std::mem::ManuallyDrop::new(S));
    drop(manual1);

    // FIXME: this line is not linted, though it should be
    f(manual2);

    // These lines will drop `S` and should be okay.
    unsafe {
        std::mem::ManuallyDrop::drop(&mut std::mem::ManuallyDrop::new(S));
        std::mem::ManuallyDrop::drop(&mut manual3);
        g(&mut manual4);
    }
}
