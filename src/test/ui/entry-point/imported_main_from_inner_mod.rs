// run-pass
#![feature(imported_main)]

pub mod foo {
    pub fn bar() {
        println!("Hello world!");
    }
}
use foo::bar as main;
