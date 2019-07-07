fn main() {
    let a = Vec::new();
    match a {
        [1, tail @ .., tail @ ..] => {},
        //~^ ERROR identifier `tail` is bound more than once in the same pattern
        //~| ERROR subslice patterns are unstable
        //~| ERROR subslice patterns are unstable
        //~| ERROR `..` can only be used once per slice pattern
        //~| ERROR expected an array or slice, found `std::vec::Vec<_>`
        _ => ()
    }
}
