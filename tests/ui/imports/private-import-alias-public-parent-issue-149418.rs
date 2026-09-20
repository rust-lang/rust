//@ run-rustfix
//@ edition: 2024

// Private import aliases should suggest their source through an accessible parent module or
// re-export.

#![allow(unused_imports, dead_code)]

mod accessible_parent {
    use self::fruits::PEAR as fruit;

    pub mod fruits {
        pub const PEAR: &str = "Pear";
    }
}

mod accessible_parent_without_self {
    use fruits::PEAR as fruit;

    pub mod fruits {
        pub const PEAR: &str = "Pear";
    }
}

mod public_reexport {
    mod private {
        pub struct Item;
    }

    pub use self::private::Item;
}

mod nested_alias {
    pub mod inner {
        use super::super::public_reexport::Item as Alias;
    }
}

mod absolute_alias {
    use crate::public_reexport::Item as Alias;
}

mod hidden_reexport {
    mod private {
        pub use crate::public_reexport::Item;
    }

    pub mod inner {
        use super::private::Item as Alias;
    }
}

mod use_site {
    fn check() {
        let _: super::nested_alias::inner::Alias;
        //~^ ERROR struct import `Alias` is private
        // Keep pointing at a re-export when the original import path is accessible here.
        let _: super::absolute_alias::Alias;
        //~^ ERROR struct import `Alias` is private
        // Finding another public re-export does not make `hidden_reexport::private` accessible.
        let _: super::hidden_reexport::inner::Alias;
        //~^ ERROR struct import `Alias` is private
    }
}

fn main() {
    let _ = accessible_parent::fruit;
    //~^ ERROR constant import `fruit` is private
    let _ = accessible_parent_without_self::fruit;
    //~^ ERROR constant import `fruit` is private
}

mod crate_visible {
    pub(crate) struct Item;

    pub mod nested {
        use crate::crate_visible::Item as Alias;
    }
}

fn check_crate_visible() {
    // A crate-visible definition can be recommended from within the same crate.
    let _: crate_visible::nested::Alias;
    //~^ ERROR struct import `Alias` is private
}

mod inaccessible_import {
    use crate::public_reexport::Item;

    pub mod nested {
        use super::Item as Alias;
    }
}

mod accessible_import {
    use crate::public_reexport::Item;

    mod nested {
        use super::Item as Alias;
    }

    mod use_site {
        fn check() {
            // The private import in our parent module is accessible here.
            let _: super::nested::Alias;
            //~^ ERROR struct import `Alias` is private
            // `super::Item` resolves here, but not through `inaccessible_import::Item`.
            // Do not recommend that inaccessible binding merely because its `Res` matches.
            let _: crate::inaccessible_import::nested::Alias;
            //~^ ERROR struct import `Alias` is private
        }
    }
}
