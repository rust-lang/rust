//@ edition: 2015
//@ run-rustfix

#![deny(absolute_paths_not_starting_with_crate)]
#![allow(unused_imports)]
#![allow(dead_code)]

pub(crate) mod foo {
    pub(crate) mod bar {
        pub(crate) mod baz { }
        pub(crate) mod baz1 { }

        pub(crate) struct XX;
    }
}

use foo::{bar::{baz::{}}};
//~^ ERROR absolute paths must start with
//~| WARN this is accepted in the current edition

use foo::{bar::{XX, baz::{}}};
//~^ ERROR absolute paths must start with
//~| WARN this is accepted in the current edition
//~| ERROR absolute paths must start with
//~| WARN this is accepted in the current edition

use foo::{bar::{baz::{}, baz1::{}}};
//~^ ERROR absolute paths must start with
//~| WARN this is accepted in the current edition
//~| ERROR absolute paths must start with
//~| WARN this is accepted in the current edition

fn main() {
}
