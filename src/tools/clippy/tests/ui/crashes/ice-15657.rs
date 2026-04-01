//@check-pass
#![warn(clippy::len_zero)]

pub struct S1;
pub struct S2;

impl S1 {
    pub fn len(&self) -> S2 {
        S2
    }
}
