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

fn main() {}
