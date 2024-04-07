trait Fun {
    pub fn test() {}
    //~^ ERROR visibility qualifiers are not permitted here
    //~| NOTE trait items always share the visibility of their trait
    //~| HELP remove the qualifier
}

fn main() {}
