#[cfg(any(cpass1))]
pub(super) fn print() {
    println!("a");
}

#[cfg(any(cpass2, cpass3))]
pub(super) fn print() {
    println!("b");
}
