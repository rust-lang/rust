fn main () {
    let sr: Vec<(u32, _, _) = vec![];
    //~^ ERROR expected one of `,` or `>`, found `=`
    //~| ERROR expected value, found struct `Vec`
    //~| ERROR mismatched types
    //~| ERROR invalid left-hand side expression
    //~| ERROR expected expression, found reserved identifier `_`
    let sr2: Vec<(u32, _, _)> = sr.iter().map(|(faction, th_sender, th_receiver)| {}).collect();
    //~^ ERROR no method named `iter` found for type `()` in the current scope
}
