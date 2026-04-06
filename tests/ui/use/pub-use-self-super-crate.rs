mod foo {
    pub use self as this;
    //~^ ERROR `self` is only public within the crate, and cannot be re-exported outside

    pub mod bar {
        pub use super as parent;
        //~^ ERROR `super` is only public within the crate, and cannot be re-exported outside
        pub use self::super as parent2;
        //~^ ERROR `super` is only public within the crate, and cannot be re-exported outside
        pub use super::{self as parent3};
        //~^ ERROR `super` is only public within the crate, and cannot be re-exported outside
        pub use self::{super as parent4};
        //~^ ERROR `super` is only public within the crate, and cannot be re-exported outside

        pub use crate as root;
        pub use crate::{self as root2};
        pub use super::super as root3;
    }
}

pub use foo::*;
pub use foo::bar::*;

pub fn main() {}
