#![crate_name="crate_method_reexport_grrrrrrr2"]

pub use name_pool::add;

pub mod name_pool {
    pub type name_pool = ();

    pub trait add {
        fn add(&self, s: String);
    }

    impl add for name_pool {
        fn add(&self, _s: String) {
        }
    }
}

pub mod rust {
    pub use crate::name_pool::add;

    pub type rt = Box<()>;

    pub trait cx {
        fn cx(&self);
    }

    impl cx for rt {
        fn cx(&self) {
        }
    }
}
