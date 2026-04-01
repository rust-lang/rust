//@ edition:2021
macro_rules! a {
    ( ) => {
        impl<'b> c for d {
            e::<f'g> //~ ERROR prefix `f` is unknown
        }
    };
}
fn main() {}
