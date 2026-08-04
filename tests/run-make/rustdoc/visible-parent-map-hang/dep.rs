#![crate_type = "lib"]
#![allow(dead_code)]

pub trait Tr { fn x(); }
pub trait Tr2 { fn x(); }

#[doc(hidden)]
pub mod outer {
        #[doc(hidden)]
        pub mod l0 {
            #[doc(hidden)] pub mod s0 { pub use super::l1; }
            #[doc(hidden)] pub mod s1 { pub use super::l1; }
            #[doc(hidden)] pub mod s2 { pub use super::l1; }
            #[doc(hidden)] pub mod s3 { pub use super::l1; }
            #[doc(hidden)] pub mod s4 { pub use super::l1; }
            #[doc(hidden)] pub mod s5 { pub use super::l1; }
            #[doc(hidden)] pub mod s6 { pub use super::l1; }
            #[doc(hidden)] pub mod s7 { pub use super::l1; }
        #[doc(hidden)]
        pub mod l1 {
            #[doc(hidden)] pub mod s0 { pub use super::l2; }
            #[doc(hidden)] pub mod s1 { pub use super::l2; }
            #[doc(hidden)] pub mod s2 { pub use super::l2; }
            #[doc(hidden)] pub mod s3 { pub use super::l2; }
            #[doc(hidden)] pub mod s4 { pub use super::l2; }
            #[doc(hidden)] pub mod s5 { pub use super::l2; }
            #[doc(hidden)] pub mod s6 { pub use super::l2; }
            #[doc(hidden)] pub mod s7 { pub use super::l2; }
        #[doc(hidden)]
        pub mod l2 {
            #[doc(hidden)] pub mod s0 { pub use super::l3; }
            #[doc(hidden)] pub mod s1 { pub use super::l3; }
            #[doc(hidden)] pub mod s2 { pub use super::l3; }
            #[doc(hidden)] pub mod s3 { pub use super::l3; }
            #[doc(hidden)] pub mod s4 { pub use super::l3; }
            #[doc(hidden)] pub mod s5 { pub use super::l3; }
            #[doc(hidden)] pub mod s6 { pub use super::l3; }
            #[doc(hidden)] pub mod s7 { pub use super::l3; }
        #[doc(hidden)]
        pub mod l3 {
            #[doc(hidden)] pub mod s0 { pub use super::l4; }
            #[doc(hidden)] pub mod s1 { pub use super::l4; }
            #[doc(hidden)] pub mod s2 { pub use super::l4; }
            #[doc(hidden)] pub mod s3 { pub use super::l4; }
            #[doc(hidden)] pub mod s4 { pub use super::l4; }
            #[doc(hidden)] pub mod s5 { pub use super::l4; }
            #[doc(hidden)] pub mod s6 { pub use super::l4; }
            #[doc(hidden)] pub mod s7 { pub use super::l4; }
        #[doc(hidden)]
        pub mod l4 {
            #[doc(hidden)] pub mod s0 { pub use super::l5; }
            #[doc(hidden)] pub mod s1 { pub use super::l5; }
            #[doc(hidden)] pub mod s2 { pub use super::l5; }
            #[doc(hidden)] pub mod s3 { pub use super::l5; }
            #[doc(hidden)] pub mod s4 { pub use super::l5; }
            #[doc(hidden)] pub mod s5 { pub use super::l5; }
            #[doc(hidden)] pub mod s6 { pub use super::l5; }
            #[doc(hidden)] pub mod s7 { pub use super::l5; }
        #[doc(hidden)]
        pub mod l5 {
            #[doc(hidden)] pub mod s0 { pub use super::l6; }
            #[doc(hidden)] pub mod s1 { pub use super::l6; }
            #[doc(hidden)] pub mod s2 { pub use super::l6; }
            #[doc(hidden)] pub mod s3 { pub use super::l6; }
            #[doc(hidden)] pub mod s4 { pub use super::l6; }
            #[doc(hidden)] pub mod s5 { pub use super::l6; }
            #[doc(hidden)] pub mod s6 { pub use super::l6; }
            #[doc(hidden)] pub mod s7 { pub use super::l6; }
        #[doc(hidden)]
        pub mod l6 {
            #[doc(hidden)] pub mod s0 { pub use super::l7; }
            #[doc(hidden)] pub mod s1 { pub use super::l7; }
            #[doc(hidden)] pub mod s2 { pub use super::l7; }
            #[doc(hidden)] pub mod s3 { pub use super::l7; }
            #[doc(hidden)] pub mod s4 { pub use super::l7; }
            #[doc(hidden)] pub mod s5 { pub use super::l7; }
            #[doc(hidden)] pub mod s6 { pub use super::l7; }
            #[doc(hidden)] pub mod s7 { pub use super::l7; }
        #[doc(hidden)]
        pub mod l7 {
            #[doc(hidden)] pub mod s0 { pub use super::l8; }
            #[doc(hidden)] pub mod s1 { pub use super::l8; }
            #[doc(hidden)] pub mod s2 { pub use super::l8; }
            #[doc(hidden)] pub mod s3 { pub use super::l8; }
            #[doc(hidden)] pub mod s4 { pub use super::l8; }
            #[doc(hidden)] pub mod s5 { pub use super::l8; }
            #[doc(hidden)] pub mod s6 { pub use super::l8; }
            #[doc(hidden)] pub mod s7 { pub use super::l8; }
        #[doc(hidden)]
        pub mod l8 {
            #[doc(hidden)] pub mod s0 { pub use super::l9; }
            #[doc(hidden)] pub mod s1 { pub use super::l9; }
            #[doc(hidden)] pub mod s2 { pub use super::l9; }
            #[doc(hidden)] pub mod s3 { pub use super::l9; }
            #[doc(hidden)] pub mod s4 { pub use super::l9; }
            #[doc(hidden)] pub mod s5 { pub use super::l9; }
            #[doc(hidden)] pub mod s6 { pub use super::l9; }
            #[doc(hidden)] pub mod s7 { pub use super::l9; }
        #[doc(hidden)]
        pub mod l9 {
            #[doc(hidden)] pub mod s0 { pub use super::l10; }
            #[doc(hidden)] pub mod s1 { pub use super::l10; }
            #[doc(hidden)] pub mod s2 { pub use super::l10; }
            #[doc(hidden)] pub mod s3 { pub use super::l10; }
            #[doc(hidden)] pub mod s4 { pub use super::l10; }
            #[doc(hidden)] pub mod s5 { pub use super::l10; }
            #[doc(hidden)] pub mod s6 { pub use super::l10; }
            #[doc(hidden)] pub mod s7 { pub use super::l10; }
        #[doc(hidden)]
        pub mod l10 {
            #[doc(hidden)] pub mod s0 { pub use super::l11; }
            #[doc(hidden)] pub mod s1 { pub use super::l11; }
            #[doc(hidden)] pub mod s2 { pub use super::l11; }
            #[doc(hidden)] pub mod s3 { pub use super::l11; }
            #[doc(hidden)] pub mod s4 { pub use super::l11; }
            #[doc(hidden)] pub mod s5 { pub use super::l11; }
            #[doc(hidden)] pub mod s6 { pub use super::l11; }
            #[doc(hidden)] pub mod s7 { pub use super::l11; }
        #[doc(hidden)]
        pub mod l11 {
            #[doc(hidden)] pub mod s0 { pub use super::leaf::S; }
            #[doc(hidden)] pub mod s1 { pub use super::leaf::S; }
            #[doc(hidden)] pub mod s2 { pub use super::leaf::S; }
            #[doc(hidden)] pub mod s3 { pub use super::leaf::S; }
            #[doc(hidden)] pub mod s4 { pub use super::leaf::S; }
            #[doc(hidden)] pub mod s5 { pub use super::leaf::S; }
            #[doc(hidden)] pub mod s6 { pub use super::leaf::S; }
            #[doc(hidden)] pub mod s7 { pub use super::leaf::S; }
            #[doc(hidden)] pub mod leaf { pub struct S; }
        }
        }
        }
        }
        }
        }
        }
        }
        }
        }
        }
        }
}

