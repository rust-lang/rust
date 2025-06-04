fn test(s: &Self::Id) {
//~^ ERROR cannot find item `Self` in this scope
//~| NOTE `Self` is only available in impls, traits, and type definitions
    match &s[0..3] {}
}

fn main() {}
