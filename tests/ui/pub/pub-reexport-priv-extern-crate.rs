extern crate core;
pub use core as reexported_core; //~ ERROR `core` is private and cannot be re-exported
                                 //~^ WARN this was previously accepted

mod foo1 {
    extern crate core;
    pub use self::core as core2; //~ ERROR extern crate `core` is private and cannot be re-exported
                                 //~^ WARN this was previously accepted
}

mod foo2 {
    use crate::foo1::core; //~ ERROR crate import `core` is private
    pub mod bar {
        extern crate core;
    }
}

mod baz {
    pub use crate::foo2::bar::core; //~ ERROR crate import `core` is private
}

fn main() {
    // Check that `foo1::core2` has the reexport's visibility and is accessible.
    foo1::core2::mem::drop(());
}
