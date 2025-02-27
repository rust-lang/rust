#![warn(clippy::module_inception)]

pub mod foo2 {
    pub mod bar2 {
        pub mod bar2 {
            //~^ module_inception

            pub mod foo2 {}
        }
        pub mod foo2 {}
    }
    pub mod foo2 {
        //~^ module_inception

        pub mod bar2 {}
    }
}

mod foo {
    mod bar {
        mod bar {
            //~^ module_inception

            mod foo {}
        }
        mod foo {}
    }
    mod foo {
        //~^ module_inception

        mod bar {}
    }
}

// No warning. See <https://github.com/rust-lang/rust-clippy/issues/1220>.
mod bar {
    #[allow(clippy::module_inception)]
    mod bar {}
}

fn main() {}
