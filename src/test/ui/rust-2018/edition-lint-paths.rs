// aux-build:edition-lint-paths.rs
// run-rustfix

#![feature(rust_2018_preview)]
#![deny(absolute_paths_not_starting_with_crate)]
#![allow(unused)]

extern crate edition_lint_paths;

pub mod foo {
    use edition_lint_paths;
    use bar::Bar;
    //~^ ERROR absolute
    //~| WARN this is accepted in the current edition
    //~| ERROR absolute
    //~| WARN this is accepted in the current edition

    use super::bar::Bar2;
    use crate::bar::Bar3;

    use bar;
    //~^ ERROR absolute
    //~| WARN this is accepted in the current edition

    use crate::bar as something_else;

    use {main, Bar as SomethingElse};
    //~^ ERROR absolute
    //~| WARN this is accepted in the current edition
    //~| ERROR absolute
    //~| WARN this is accepted in the current edition
    //~| ERROR absolute
    //~| WARN this is accepted in the current edition

    use crate::{main as another_main, Bar as SomethingElse2};

    pub fn test() {}

    pub trait SomeTrait {}
}

use bar::Bar;
//~^ ERROR absolute
//~| WARN this is accepted in the current edition
//~| ERROR absolute
//~| WARN this is accepted in the current edition

pub mod bar {
    use edition_lint_paths as foo;
    pub struct Bar;
    pub type Bar2 = Bar;
    pub type Bar3 = Bar;
}

mod baz {
    use *;
    //~^ ERROR absolute
    //~| WARN this is accepted in the current edition
}

impl ::foo::SomeTrait for u32 {}
//~^ ERROR absolute
//~| WARN this is accepted in the current edition
//~| ERROR absolute
//~| WARN this is accepted in the current edition

fn main() {
    let x = ::bar::Bar;
    //~^ ERROR absolute
    //~| WARN this is accepted in the current edition

    let x = bar::Bar;
    let x = crate::bar::Bar;
    let x = self::bar::Bar;
    foo::test();

    {
        use edition_lint_paths as bar;
        edition_lint_paths::foo();
        bar::foo();
        ::edition_lint_paths::foo();
    }
}
