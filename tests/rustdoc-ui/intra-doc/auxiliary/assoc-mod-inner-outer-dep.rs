#[derive(Clone)]
pub struct Struct;

pub mod dep_outer1 {
    /// [crate::Struct::clone]
    pub mod inner {}
}

pub mod dep_outer2 {
    //! [crate::Struct::clone]
}
