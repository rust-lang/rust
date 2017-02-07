#![feature(plugin)]
#![plugin(clippy)]
#![deny(module_inception)]

mod foo {
    mod bar {
        mod bar { //~ ERROR module has the same name as its containing module
            mod foo {}
        }
        mod foo {}
    }
    mod foo { //~ ERROR module has the same name as its containing module
        mod bar {}
    }
}

// No warning. See <https://github.com/Manishearth/rust-clippy/issues/1220>.
mod bar {
    #[allow(module_inception)]
    mod bar {
    }
}

fn main() {}
