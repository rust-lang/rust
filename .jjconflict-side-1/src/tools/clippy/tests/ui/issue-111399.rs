//@ check-pass

#![feature(inherent_associated_types)]
#![expect(incomplete_features)]

// Check that rustc doesn't crash on the trait bound `Self::Ty: std::marker::Freeze`.

pub struct Struct;

impl Struct {
    pub type Ty = usize;
    pub const CT: Self::Ty = 42;
}

fn main() {}
