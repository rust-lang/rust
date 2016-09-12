#![feature(plugin)]
#![plugin(clippy)]
#![deny(module_inception)]

mod foo {
    mod bar {
        pub mod bar { //~ ERROR item has the same name as its containing module
            mod foo {}
        }
        mod foo {}
    }
    pub mod foo { //~ ERROR item has the same name as its containing module
        mod bar {}
    }
}

mod cake {
    mod cake {
        // no error, since module is not public
    }
}

// No warning. See <https://github.com/Manishearth/rust-clippy/issues/1220>.
mod bar {
    #[allow(module_inception)]
    pub mod bar {
    }
}

fn main() {}
