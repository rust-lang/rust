#![feature(type_alias_impl_trait)]
#![allow(incomplete_features)]
pub mod foo {
    type MainFn = impl Fn();

    fn bar() {}
    #[define_opaque(MainFn)]
    const fn def() -> MainFn {
        bar
    }
    pub const BAR: MainFn = def();
}

use foo::BAR as main; //~ ERROR `main` function not found in crate
