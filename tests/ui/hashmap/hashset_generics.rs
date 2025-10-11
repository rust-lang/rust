use std::collections::HashSet;

#[derive(PartialEq)]
//~^ NOTE in this expansion of
//~| NOTE in this expansion of
//~| NOTE in this expansion of
pub struct MyStruct {
    pub parameters: HashSet<String, String>,
    //~^ NOTE the foreign item type
    //~| ERROR binary operation
}

fn main() {}
