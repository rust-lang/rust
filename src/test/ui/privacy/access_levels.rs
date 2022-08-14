#![feature(rustc_attrs)]

#[rustc_access_level] mod outer { //~ ERROR None
    #[rustc_access_level] pub mod inner { //~ ERROR Some(Exported)
        #[rustc_access_level]
        extern "C" { //~ ERROR Some(Exported)
            #[rustc_access_level] static a: u8; //~ ERROR None
            #[rustc_access_level] pub fn b(); //~ ERROR Some(Exported)
        }
        #[rustc_access_level]
        pub trait Trait { //~ ERROR Some(Exported)
            #[rustc_access_level] const A: i32; //~ ERROR Some(Exported)
            #[rustc_access_level] type B; //~ ERROR Some(Exported)
        }

        #[rustc_access_level]
        pub struct Struct { //~ ERROR Some(Exported)
            #[rustc_access_level] a: u8, //~ ERROR None
            #[rustc_access_level] pub b: u8, //~ ERROR Some(Exported)
        }

        #[rustc_access_level]
        pub union Union { //~ ERROR Some(Exported)
            #[rustc_access_level] a: u8, //~ ERROR None
            #[rustc_access_level] pub b: u8, //~ ERROR Some(Exported)
        }

        #[rustc_access_level]
        pub enum Enum { //~ ERROR Some(Exported)
            #[rustc_access_level] A( //~ ERROR Some(Exported)
                #[rustc_access_level] Struct, //~ ERROR Some(Exported)
                #[rustc_access_level] Union,  //~ ERROR Some(Exported)
            ),
        }
    }

    #[rustc_access_level] macro_rules! none_macro { //~ ERROR None
        () => {};
    }

    #[macro_export]
    #[rustc_access_level] macro_rules! public_macro { //~ ERROR Some(Public)
        () => {};
    }
}

pub use outer::inner;

fn main() {}
