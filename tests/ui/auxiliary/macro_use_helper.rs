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
    pub use macro_rules::try_err;

    // ITEM
    #[macro_export]
    macro_rules! inner_mod {
        () => {
            #[allow(dead_code)]
            pub struct Tardis;
        };
    }
}

// EXPR
#[macro_export]
macro_rules! function {
    () => {
        if true {
        } else {
        }
    };
}

// TYPE
#[macro_export]
macro_rules! ty_mac {
    () => {
        Vec<u8>
    };
}

mod extern_exports {
    pub(super) mod private_inner {
        #[macro_export]
        macro_rules! pub_in_private {
            ($name:ident) => {
                let $name = String::from("secrets and lies");
            };
        }
    }
}
