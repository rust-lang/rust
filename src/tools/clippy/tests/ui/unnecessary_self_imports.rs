#![warn(clippy::unnecessary_self_imports)]

use std::collections::hash_map::{self, *};
use std::fs::{self as alias};
//~^ unnecessary_self_imports
use std::io::{self, Read};
use std::rc::{self};
//~^ unnecessary_self_imports

// https://github.com/rust-lang/rust-clippy/issues/17652
fn nested_imports() {
    mod a {
        pub mod b {
            pub mod c {}
        }
        pub mod d {
            pub mod e {
                pub mod f {}
            }
        }
    }

    #[rustfmt::skip]
    use a::{
        b::{self},
        //~^ unnecessary_self_imports
        b::{
            c::{self},
            //~^ unnecessary_self_imports
        },
        d::{
            // don't lint, as there are other imports in this group
            self,
            e::{self},
            //~^ unnecessary_self_imports
            e::{
                f::{self}
                //~^ unnecessary_self_imports
            },
        },
    };
}

fn main() {}
