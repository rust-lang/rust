#![warn(clippy::module_inception)]

pub mod foo2 {
    pub mod bar2 {
        pub mod bar2 {
            //~^ ERROR: module has the same name as its containing module
            //~| NOTE: `-D clippy::module-inception` implied by `-D warnings`
            pub mod foo2 {}
        }
        pub mod foo2 {}
    }
    pub mod foo2 {
        //~^ ERROR: module has the same name as its containing module
        pub mod bar2 {}
    }
}

mod foo {
    mod bar {
        mod bar {
            //~^ ERROR: module has the same name as its containing module
            mod foo {}
        }
        mod foo {}
    }
    mod foo {
        //~^ ERROR: module has the same name as its containing module
        mod bar {}
    }
}

// No warning. See <https://github.com/rust-lang/rust-clippy/issues/1220>.
mod bar {
    #[allow(clippy::module_inception)]
    mod bar {}
}

fn main() {}
