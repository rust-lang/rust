fn test(s: &Self::Id) {
//~^ ERROR failed to resolve: `Self` is only available in impls, traits, and type definitions
    match &s[0..3] {}
}

fn main() {}
