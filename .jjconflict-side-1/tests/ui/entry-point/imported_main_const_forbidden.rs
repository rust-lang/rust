pub mod foo {
    pub const BAR: usize = 42;
}

use foo::BAR as main; //~ ERROR `main` function not found in crate
