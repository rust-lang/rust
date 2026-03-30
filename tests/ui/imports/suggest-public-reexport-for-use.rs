//@ edition:2021

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

// From a sibling module, the suggestion should use `super::`.
mod sibling {
    use crate::outer::inner::MyStruct; //~ ERROR module `inner` is private
}

// From a deeply nested module, the suggestion should keep the full path.
mod deep {
    mod nested {
        use crate::outer::inner::MyStruct; //~ ERROR module `inner` is private
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
