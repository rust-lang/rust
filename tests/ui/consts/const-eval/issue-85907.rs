const fn hey() -> usize {
    panic!(123); //~ ERROR argument to `panic!()` in a const context must have type `&str`
}

fn main() {
    let _: [u8; hey()] = todo!();
}
