// A single lifetime is not parsed as a type.
// `ty` matcher in particular doesn't accept a single lifetime

macro_rules! m {
    ($t: ty) => ( let _: $t; )
}

fn main() {
    m!('static); //~ ERROR expected type, found `'static`
}
