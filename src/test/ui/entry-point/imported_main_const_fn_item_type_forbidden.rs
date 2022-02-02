#![feature(imported_main)]
#![feature(type_alias_impl_trait)]
#![allow(incomplete_features)]
pub mod foo {
    type MainFn = impl Fn();
    //~^ ERROR could not find defining uses

    fn bar() {}
    pub const BAR: MainFn = bar;
    //~^ ERROR mismatched types [E0308]
}

use foo::BAR as main; //~ ERROR `main` function not found in crate
