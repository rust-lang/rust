pub struct T;
pub struct U;
pub struct Const<const N: usize>;
pub const N: usize = 1;

pub trait Tr {
    type Assoc;
}

impl Tr for T {
    type Assoc = U;
}
