// check-pass

mod a {
    pub trait P {}
}
pub use a::*;

mod b {
    #[derive(Clone)]
    pub enum P {
        A
    }
}
pub use b::P;

mod c {
    use crate::*;
    pub struct _S(Vec<P>);
}

fn main() {}
