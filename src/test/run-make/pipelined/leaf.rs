#![crate_type = "rlib"]

pub static FOO: &str = "this is a static";

#[derive(Debug)]
pub struct Plain {
    pub a: u32,
    pub b: u32,
}

#[derive(Debug)]
pub struct GenericStruct<A> {
    pub a: A,
    pub b: Option<A>,
}

pub fn simple(a: u32, b: u32) -> u32 {
    let c = a + b;
    println!("simple {} + {} => {}", a, b, c);
    c
}

pub fn generic<D: std::fmt::Debug>(d: D) {
    println!("generically printing {:?}", d);
}
