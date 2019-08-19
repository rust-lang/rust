// run-pass

pub trait Stage: Sync {}

pub enum Enum {
    A,
    B,
}

impl Stage for Enum {}

pub static ARRAY: [(&dyn Stage, &str); 1] = [
    (&Enum::A, ""),
];

fn main() {}
