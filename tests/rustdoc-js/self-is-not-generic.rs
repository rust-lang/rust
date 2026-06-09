pub trait Combine {
    fn combine(&self, other: &Self) -> Self;
}

pub struct Thing;

impl Combine for Thing {
    fn combine(&self, other: &Self) -> Self {
        Self
    }
}
