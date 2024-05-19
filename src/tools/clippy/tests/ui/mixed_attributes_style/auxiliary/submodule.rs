//! Module level doc

#![allow(dead_code)]

#[allow(unused)]
//~^ ERROR: item has both inner and outer attributes
mod foo {
    #![allow(dead_code)]
}
