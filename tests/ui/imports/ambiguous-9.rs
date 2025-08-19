// https://github.com/rust-lang/rust/pull/113099#issuecomment-1638206152

pub mod dsl {
    mod range {
        pub fn date_range() {}
    }
    pub use self::range::*; //~ WARNING ambiguous glob re-exports
    use super::prelude::*;
}

pub mod prelude {
    mod t {
      pub fn date_range() {}
    }
    pub use self::t::*; //~ WARNING ambiguous glob re-exports
    pub use super::dsl::*;
}

use dsl::*;
use prelude::*;

fn main() {
    date_range();
    //~^ ERROR `date_range` is ambiguous
    //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| ERROR `date_range` is ambiguous
    //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}
