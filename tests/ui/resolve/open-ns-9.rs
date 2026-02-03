// Tests that namespaced crate names work inside macros.
//@ aux-crate: my_api::utils=open-ns-my_api_utils.rs
//@ compile-flags: -Z namespaced-crates
//@ edition: 2024
//@ build-pass

macro_rules! define {
    () => {
        pub mod my_api {
            pub mod utils {
                pub fn get_u32() -> u32 {
                    2
                }
            }

            pub mod unique {
                pub fn get_u32() -> u32 {
                    2
                }
            }
        }
    };
}

fn main() {
    define!();
    let res1 = my_api::unique::get_u32();
    assert_eq!(res1, 2);
}
