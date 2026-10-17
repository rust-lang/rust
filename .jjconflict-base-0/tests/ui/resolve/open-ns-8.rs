// Tests that a macro-generated item has higher precendence than a namespaced crate
//@ aux-crate: my_api::utils=open-ns-my_api_utils.rs
//@ compile-flags: -Z namespaced-crates
//@ edition: 2024
//@ check-pass

macro_rules! define {
    () => {
        pub mod my_api {
            pub mod utils {
                pub fn get_u32() -> u32 {
                    2
                }
            }
        }
    };
}

fn main() {
    define!();
    let res = my_api::utils::get_u32();
    assert_eq!(res, 2);
}
