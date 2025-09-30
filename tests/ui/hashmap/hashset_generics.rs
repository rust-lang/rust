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

fn main() {
    let h1 = HashSet::<usize, usize>::with_hasher(0);
    h1.insert(1);
    //~^ ERROR its trait bounds were not satisfied
    //~| NOTE the following trait bounds
}
