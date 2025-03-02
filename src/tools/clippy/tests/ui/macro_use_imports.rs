//@aux-build:macro_rules.rs
//@aux-build:macro_use_helper.rs
//@aux-build:proc_macro_derive.rs

//@ignore-bitwidth: 32

#![allow(unused_imports, unreachable_code, unused_variables, dead_code, unused_attributes)]
#![allow(clippy::single_component_path_imports)]
#![warn(clippy::macro_use_imports)]

#[macro_use]
extern crate macro_use_helper as mac;

#[macro_use]
extern crate proc_macro_derive as mini_mac;

mod a {
    #[macro_use]
    //~^ macro_use_imports
    use mac;
    #[macro_use]
    //~^ macro_use_imports
    use mini_mac;
    #[macro_use]
    //~^ macro_use_imports
    use mac::inner;
    #[macro_use]
    //~^ macro_use_imports
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
        inner::mut_mut!();
        nested::string_add!();
    }
}

// issue #7015, ICE due to calling `module_children` with local `DefId`
#[macro_use]
use a as b;

fn main() {}
