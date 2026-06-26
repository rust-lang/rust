// Regression test for issue #156264

//@ check-pass

mod m_pub {
    pub struct S {}
}

mod m_crate {
    pub(crate) use crate::m_pub::S;
}

pub(crate) use m_crate::*;
pub use m_pub::*;

fn main() {}
