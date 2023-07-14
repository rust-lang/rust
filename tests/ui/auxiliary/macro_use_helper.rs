//@aux-build:macro_rules.rs

extern crate macro_rules;

// STMT
#[macro_export]
macro_rules! pub_macro {
    () => {
        let _ = "hello Mr. Vonnegut";
    };
}

pub mod inner {
    pub use super::*;

    // RE-EXPORT
    // this will stick in `inner` module
    pub use macro_rules::{mut_mut, try_err};

    pub mod nested {
        pub use macro_rules::string_add;
    }

    // ITEM
    #[macro_export]
    macro_rules! inner_mod_macro {
        () => {
            #[allow(dead_code)]
            pub struct Tardis;
        };
    }
}

// EXPR
#[macro_export]
macro_rules! function_macro {
    () => {
        if true {
        } else {
        }
    };
}

// TYPE
#[macro_export]
macro_rules! ty_macro {
    () => {
        Vec<u8>
    };
}

mod extern_exports {
    pub(super) mod private_inner {
        #[macro_export]
        macro_rules! pub_in_private_macro {
            ($name:ident) => {
                let $name = String::from("secrets and lies");
            };
        }
    }
}
