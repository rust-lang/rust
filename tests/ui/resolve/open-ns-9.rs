//@ aux-crate: my_api::utils=open-ns-my_api_utils.rs
//@ compile-flags: -Z namespaced-crates
//@ edition: 2024

use my_api::utils::get_u32;
//~^ ERROR `my_api` is ambiguous [E0659]

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

define!();

fn main() {
    let val = get_u32();
    assert_eq!(val, 2);
}
