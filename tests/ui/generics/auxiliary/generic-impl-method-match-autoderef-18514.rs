#![crate_type = "rlib"]

pub trait Tr {
    fn tr(&self);
}

pub struct St<V>(pub Vec<V>);

impl<V> Tr for St<V> {
    fn tr(&self) {
        match self {
            &St(ref v) => {
                v.iter();
            }
        }
    }
}
