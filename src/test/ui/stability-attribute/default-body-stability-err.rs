// aux-build:default_body.rs
#![crate_type = "lib"]

extern crate default_body;

use default_body::{Equal, JustTrait};

struct Type;

impl JustTrait for Type {}
//~^ ERROR use of unstable library feature 'fun_default_body'
//~| ERROR use of unstable library feature 'constant_default_body'

impl Equal for Type {
    //~^ ERROR use of unstable library feature 'eq_default_body'
    fn neq(&self, other: &Self) -> bool {
        false
    }
}
