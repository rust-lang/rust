fn main() {
    match &[1, 2, 3][..] {
        [1, rest..] => println!("{rest}"),
        //~^ ERROR cannot find value `rest` in this scope
        //~| ERROR cannot find value `rest` in this scope
        //~| ERROR `X..` patterns in slices are experimental
        _ => {}
    }
    match &[4, 5, 6][..] {
        [] => {}
        [_, ..tail] => println!("{tail}"),
        //~^ ERROR cannot find value `tail` in this scope
        //~| ERROR cannot find value `tail` in this scope
    }
    match &[7, 8, 9][..] {
        [] => {}
        [_, ...tail] => println!("{tail}"),
        //~^ ERROR cannot find value `tail` in this scope
        //~| ERROR cannot find value `tail` in this scope
        //~| ERROR range-to patterns with `...` are not allowed
    }
}
