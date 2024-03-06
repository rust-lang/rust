//@ run-pass
#![allow(dead_code)]
use module_of_many_things::*;
use dug::too::greedily::and::too::deep::*;

mod module_of_many_things {
    pub fn f1() { println!("f1"); }
    pub fn f2() { println!("f2"); }
    fn f3() { println!("f3"); }
    pub fn f4() { println!("f4"); }
}

mod dug {
    pub mod too {
        pub mod greedily {
            pub mod and {
                pub mod too {
                    pub mod deep {
                        pub fn nameless_fear() { println!("Boo!"); }
                        pub fn also_redstone() { println!("Whatever."); }
                    }
                }
            }
        }
    }
}


pub fn main() { f1(); f2(); f4(); nameless_fear(); also_redstone(); }
