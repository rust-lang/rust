//! Regression test for <https://github.com/rust-lang/rust/issues/115858>.
//!
//! The compiler used to suggest `crate::crate::unix::linux::system::Y`
//! (duplicating the `crate` keyword) instead of `crate::unix::linux::system::Y`.

pub mod unix {
    pub mod linux {
        pub mod utils {
            pub fn f() {
                let _x = crate::linux::system::Y;
                //~^ ERROR cannot find `linux` in `crate`
            }
        }
        pub mod system {
            pub const Y: u32 = 0;
        }
    }
}

fn main() {}
