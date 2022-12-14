// A single lifetime is not parsed as a type.
// `ty` matcher in particular doesn't accept a single lifetime

macro_rules! m {
    ($t: ty) => {
        let _: $t;
    };
}

fn main() {
    m!('static);
    //~^ ERROR lifetime in trait object type must be followed by `+`
    //~| ERROR at least one trait is required for an object type
}
