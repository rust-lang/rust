#![crate_type="lib"]
#![crate_name="issue12660aux"]

pub use my_mod::{MyStruct, my_fn};

mod my_mod {
    pub struct MyStruct;

    pub fn my_fn(my_struct: MyStruct) {
    }
}
