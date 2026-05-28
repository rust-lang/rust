use std::collections::HashSet;

#[derive(PartialEq)]
//~^ NOTE in this expansion of
//~| NOTE in this expansion of
//~| NOTE in this expansion of
pub struct MyStruct {
    pub parameters: HashSet<String, String>,
    //~^ NOTE `String` does not implement `BuildHasher`
    //~| ERROR binary operation
    //~| HELP use a HashMap
}

fn main() {
    let h1 = HashSet::<usize, usize>::with_hasher(0);
    h1.insert(1);
    //~^ ERROR its trait bounds were not satisfied
    //~| NOTE the following trait bounds
    //~| HELP use a HashMap
}
