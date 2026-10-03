//@ run-rustfix
fn main() {
    let v = &mut &mut Vec::<i32>::new();
    for _ in &mut &mut v {} //~ ERROR E0277

    let v = &mut &mut [1u8];
    for _ in &mut v {} //~ ERROR E0277
}
