//@ check-pass

#![no_std]
#![crate_type = "lib"]

macro_rules! make_prelude {
    () => {
        mod prelude {
            #[allow(unused_imports)]
            pub(crate) use core::clone::Clone;
            pub(crate) use core::default::Default;
            #[allow(unused_imports)]
            pub(crate) use core::marker::Copy;
            pub(crate) use core::prelude::v1::derive;
            pub(crate) use core::cfg;
        }
    };
}

make_prelude!();

mod generated {
    use crate::prelude::*;

    macro_rules! define_struct {
        ($(#[$attr:meta])* pub struct $name:ident;) => {
            #[::core::prelude::v1::derive(
                ::core::clone::Clone,
                ::core::marker::Copy,
                ::core::fmt::Debug,
            )]
            $(#[$attr])*
            pub struct $name;
        };
    }

    define_struct! {
        #[derive(Default)]
        pub struct Generated;
    }

    pub const CONFIGURED: bool = cfg!(unix);
}

pub use generated::{CONFIGURED, Generated};
