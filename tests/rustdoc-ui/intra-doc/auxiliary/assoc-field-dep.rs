#[derive(Clone)]
pub struct Struct;

pub mod dep_mod1 {
    pub struct Fields {
        /// [crate::Struct::clone]
        pub field: u8,
    }
}

pub mod dep_mod2 {
    pub enum Fields {
        V {
            /// [crate::Struct::clone]
            field: u8,
        },
    }
}
