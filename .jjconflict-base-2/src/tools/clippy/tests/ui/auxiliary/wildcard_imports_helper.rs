pub use crate::extern_exports::*;

pub fn extern_foo() {}
pub fn extern_bar() {}

pub struct ExternA;

pub mod inner {
    pub mod inner_for_self_import {
        pub fn inner_extern_foo() {}
        pub fn inner_extern_bar() {}
    }
}

mod extern_exports {
    pub fn extern_exported() {}
    pub struct ExternExportedStruct;
    pub enum ExternExportedEnum {
        A,
    }
}

pub mod prelude {
    pub mod v1 {
        pub struct PreludeModAnywhere;
    }
}

pub mod extern_prelude {
    pub mod v1 {
        pub struct ExternPreludeModAnywhere;
    }
}
