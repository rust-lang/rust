extern crate core;
pub use core as reexported_core;
//~^ ERROR `core` is only public within the crate, and cannot be re-exported outside

mod foo1 {
    extern crate core;
    pub use self::core as core2; //~ ERROR `core` is private, and cannot be re-exported
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
    foo1::core2::mem::drop(()); //~ ERROR crate import `core2` is private
}
