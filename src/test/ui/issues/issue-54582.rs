// run-pass

pub trait Stage: Sync {}

pub enum Enum {
    A,
    B,
}

impl Stage for Enum {}

pub static ARRAY: [(&Stage, &str); 1] = [
    (&Enum::A, ""),
];

fn main() {}
