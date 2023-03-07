#![feature(imported_main)]

mod foo {
    pub(crate) fn bar() {}
}

use foo::bar as main;
