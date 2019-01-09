#![warn(clippy::module_inception)]

mod foo {
    mod bar {
        mod bar {
            mod foo {}
        }
        mod foo {}
    }
    mod foo {
        mod bar {}
    }
}

// No warning. See <https://github.com/rust-lang/rust-clippy/issues/1220>.
mod bar {
    #[allow(clippy::module_inception)]
    mod bar {}
}

fn main() {}
