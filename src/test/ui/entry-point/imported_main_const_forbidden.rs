#![feature(imported_main)]
//~^ ERROR `main` function not found in crate
pub mod foo {
    pub const BAR: usize = 42;
}

use foo::BAR as main;
