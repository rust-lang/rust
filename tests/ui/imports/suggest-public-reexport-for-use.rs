//@ revisions: edition2015 edition2018 edition2021 edition2024
//@ [edition2015] edition:2015
//@ [edition2018] edition:2018
//@ [edition2021] edition:2021
//@ [edition2024] edition:2024

// When a `use` statement accesses an item through a private module,
// the compiler should suggest a public re-export if one exists.

mod outer {
    pub use self::inner::MyStruct;
    pub use self::inner::my_function;
    pub use self::inner::MyTrait;
    pub use self::inner::MyEnum;

    mod inner {
        pub struct MyStruct;
        pub fn my_function() {}
        pub trait MyTrait {}
        pub enum MyEnum {
            Variant,
        }
    }
}

// Accessing items through a private module should suggest the public re-export.
use outer::inner::MyStruct; //~ ERROR module `inner` is private
use outer::inner::my_function; //~ ERROR module `inner` is private
use outer::inner::MyTrait; //~ ERROR module `inner` is private
use outer::inner::MyEnum; //~ ERROR module `inner` is private

// From a sibling module, the suggestion should keep the full path
// (shortening to `super::` would not reduce the segment count here).
mod sibling {
    #[cfg(edition2015)]
    use outer::inner::MyStruct; //[edition2015]~ ERROR module `inner` is private

    #[cfg(not(edition2015))]
    use crate::outer::inner::MyStruct; //[edition2018,edition2021,edition2024]~ ERROR module `inner` is private
}

// From a deeply nested module, the suggestion should keep the full path.
mod deep {
    mod nested {
        #[cfg(edition2015)]
        use outer::inner::MyStruct; //[edition2015]~ ERROR module `inner` is private

        #[cfg(not(edition2015))]
        use crate::outer::inner::MyStruct; //[edition2018,edition2021,edition2024]~ ERROR module `inner` is private
    }
}

// Items with no public re-export should say "not publicly re-exported".
mod no_reexport {
    mod hidden {
        pub struct Secret;
    }
}

use no_reexport::hidden::Secret; //~ ERROR module `hidden` is private

fn main() {}
