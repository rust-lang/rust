//@ aux-build:offset-of-staged-api.rs

use std::mem::offset_of;

extern crate offset_of_staged_api;

use offset_of_staged_api::*;

fn main() {
    offset_of!(
        //~^ ERROR use of unstable library feature
        Unstable, //~ ERROR use of unstable library feature
        unstable
    );
    offset_of!(Stable, stable);
    offset_of!(StableWithUnstableField, unstable); //~ ERROR use of unstable library feature
    offset_of!(StableWithUnstableFieldType, stable);
    offset_of!(StableWithUnstableFieldType, stable.unstable); //~ ERROR use of unstable library feature
    offset_of!(
        //~^ ERROR use of unstable library feature
        UnstableWithStableFieldType, //~ ERROR use of unstable library feature
        unstable
    );
    offset_of!(
        //~^ ERROR use of unstable library feature
        UnstableWithStableFieldType, //~ ERROR use of unstable library feature
        unstable.stable
    );
}
