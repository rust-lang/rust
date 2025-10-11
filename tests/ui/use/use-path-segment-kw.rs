// mod x {
//     use super; // bad
//     use super as name; // good
//     use self; // bad
//     use self as name; // good
//     use crate; // bad
//     use crate as name; // good

//     mod y;
//     use y::crate; // bad
//     use crate::crate; // bad

//     // use $crate; // bad
//     // use $crate as name; // good

//     // use super::{self}; // bad
//     // use super::{self as name}; // good
//     // use crate::{self}; // bad
//     // use crate::{self as name}; // good
//     // use $crate::{self}; // bad
//     // use $crate::{self as name}; // good
// }

mod foo {
    pub mod foobar {
        pub use super as x;
    }
    pub fn bar() {
        println!("hello");
    }
}

fn main() {
    foo::foobar::x::bar();
}
