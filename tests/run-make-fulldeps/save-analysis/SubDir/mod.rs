// sub-module in a sub-directory

use sub::sub2 as msalias;
use sub::sub2;

static yy: usize = 25;

mod sub {
    pub mod sub2 {
        pub mod sub3 {
            pub fn hello() {
                println!("hello from module 3");
            }
        }
        pub fn hello() {
            println!("hello from a module");
        }

        pub struct nested_struct {
            pub field2: u32,
        }
    }
}

pub struct SubStruct {
    pub name: String
}
