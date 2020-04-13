fn main () {
    let sr: Vec<(u32, _, _) = vec![];
    //~^ ERROR expected one of `,` or `>`, found `=`
    //~| ERROR expected value, found struct `Vec`
    //~| ERROR expected value, found builtin type `u32`
    //~| ERROR mismatched types
    //~| ERROR invalid left-hand side of assignment
    //~| ERROR expected expression, found reserved identifier `_`
    //~| ERROR expected expression, found reserved identifier `_`
    let sr2: Vec<(u32, _, _)> = sr.iter().map(|(faction, th_sender, th_receiver)| {}).collect();
    //~^ ERROR a value of type `Vec<(u32, _, _)>` cannot be built
}
