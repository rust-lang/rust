// `ty` matcher accepts trait object types

macro_rules! m {
    ($t: ty) => ( let _: $t; )
}

fn main() {
    m!(Copy + Send + 'static); //~ ERROR the trait `std::marker::Copy` cannot be made into an object
    m!('static + Send);
    m!('static +); //~ ERROR at least one non-builtin trait is required for an object type
}
