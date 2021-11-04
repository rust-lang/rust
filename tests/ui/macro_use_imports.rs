// aux-build:macro_rules.rs
// aux-build:macro_use_helper.rs
// aux-build:proc_macro_derive.rs
// run-rustfix
// ignore-32bit

#![allow(unused_imports, unreachable_code, unused_variables, dead_code, unused_attributes)]
#![allow(clippy::single_component_path_imports)]
#![warn(clippy::macro_use_imports)]

#[macro_use]
extern crate macro_use_helper as mac;

#[macro_use]
extern crate proc_macro_derive as mini_mac;

mod a {
    #[macro_use]
    use mac;
    #[macro_use]
    use mini_mac;
    #[macro_use]
    use mac::inner;
    #[macro_use]
    use mac::inner::nested;

    #[derive(ClippyMiniMacroTest)]
    struct Test;

    fn test() {
        pub_macro!();
        inner_mod_macro!();
        pub_in_private_macro!(_var);
        function_macro!();
        let v: ty_macro!() = Vec::default();

        inner::try_err!();
        inner::foofoo!();
        nested::string_add!();
    }
}

// issue #7015, ICE due to calling `item_children` with local `DefId`
#[macro_use]
use a as b;

fn main() {}
