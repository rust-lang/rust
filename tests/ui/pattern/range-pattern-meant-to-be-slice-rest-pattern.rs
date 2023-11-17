fn main() {
    match &[1, 2, 3][..] {
        [1, rest..] => println!("{rest:?}"),
        //~^ ERROR cannot find value `rest` in this scope
        //~| ERROR cannot find value `rest` in this scope
        //~| ERROR `X..` patterns in slices are experimental
        _ => {}
    }
}
