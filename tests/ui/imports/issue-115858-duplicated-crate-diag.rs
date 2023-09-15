// Regression test for <https://github.com/rust-lang/rust/issues/115858>.
// It ensures that it doesn't suggest paths like `crate::crate::unix`

// edition:2018

#![crate_type = "lib"]

pub mod unix {
    pub mod linux {
        pub mod utils {
            pub fn f() {
                let x = crate::linux::system::Y;
                //~^ ERROR unresolved import
            }
        }
        pub mod system {
            pub const Y: u32 = 0;
        }
    }
}
