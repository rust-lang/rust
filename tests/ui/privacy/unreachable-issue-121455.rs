fn test(s: &Self::Id) {
//~^ ERROR cannot find item `Self`
    match &s[0..3] {}
}

fn main() {}
