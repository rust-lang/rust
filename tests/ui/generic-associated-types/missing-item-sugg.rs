//@ aux-build:missing-item-sugg.rs

extern crate missing_item_sugg;

struct Local;
impl missing_item_sugg::Foo for Local {
    //~^ ERROR not all trait items implemented, missing: `Gat`
}
//~^ HELP implement the missing item: `type Gat<T> = /* Type */ where T: std::fmt::Display;`

fn main() {}
