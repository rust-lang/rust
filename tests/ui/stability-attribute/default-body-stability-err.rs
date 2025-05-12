//@ aux-build:default_body.rs
#![crate_type = "lib"]

extern crate default_body;

use default_body::{Equal, JustTrait};

struct Type;

impl JustTrait for Type {}
//~^ ERROR not all trait items implemented, missing: `CONSTANT` [E0046]
//~| ERROR not all trait items implemented, missing: `fun` [E0046]
//~| ERROR not all trait items implemented, missing: `fun2` [E0046]

impl Equal for Type {
    //~^ ERROR not all trait items implemented, missing: `eq` [E0046]
    fn neq(&self, other: &Self) -> bool {
        false
    }
}
