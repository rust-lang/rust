//@ run-pass
#![deny(missing_docs)]
//! this tests the syntax of `thread_local!`

mod foo {
    mod bar {
        thread_local! {
            // no docs
            #[allow(unused)]
            static FOO: i32 = 42;
            /// docs
            pub static BAR: String = String::from("bar");

            // look at these restrictions!!
            pub(crate) static BAZ: usize = 0;
            pub(in crate::foo) static QUUX: usize = 0;
        }
        thread_local!(static SPLOK: u32 = 0);
    }
}

fn main() {}
